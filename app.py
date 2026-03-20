"""
a2doc Flask web application.
Bilibili audio downloader and transcriber with LLM completion and summarisation.
"""

import atexit
import json
import os
import queue
import random
import shutil
import signal
import sys
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
DATA_TEMP = DATA_DIR / "temp"
DATA_USERDATA = DATA_DIR / "userdata"

DATA_TEMP.mkdir(parents=True, exist_ok=True)
DATA_USERDATA.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Lazy imports from src/utils (they may have heavy deps like sherpa-onnx)
# ---------------------------------------------------------------------------
from src.utils.extract_url import download_bilibili_wav  # noqa: E402
from src.utils.asr import ensure_model, load_recognizer, transcribe as asr_transcribe, MODEL_TYPE  # noqa: E402
from src.utils.llm import complete_transcription, summarize_text, PROVIDERS  # noqa: E402
from src.utils.merge import merge_files, find_userdata_files  # noqa: E402

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# ASR model preloading
# ---------------------------------------------------------------------------
_recognizer = None
_model_ready = threading.Event()

def _load_model():
    global _recognizer
    try:
        ensure_model(MODEL_TYPE)
        _recognizer = load_recognizer(MODEL_TYPE)
        print("[ASR] 模型就绪")
    except Exception as exc:
        print(f"[ASR] 模型加载失败: {exc}")
    finally:
        _model_ready.set()

def _release_model():
    global _recognizer
    _recognizer = None

atexit.register(_release_model)
signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
threading.Thread(target=_load_model, daemon=True).start()

# ---------------------------------------------------------------------------
# Processing state (module-level singletons)
# ---------------------------------------------------------------------------
status_messages: list[dict] = []
_status_lock = threading.Lock()
_process_lock = threading.Lock()
_is_processing = False
_requested_bvids: list[str] = []

DEFAULT_SUMMARY_PROMPT = "请用中文总结这段视频的主要内容，列出关键观点和结论。"

def _emit(msg_type: str, text: str) -> None:
    """Append a status message visible to the SSE endpoint."""
    with _status_lock:
        status_messages.append({"type": msg_type, "text": text})

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _check_existing(bvid: str) -> dict:
    """Return existing transcript/summary paths for bvid, or None."""
    transcripts = sorted(DATA_USERDATA.glob(f"{bvid}-*-transcript.txt"))
    summaries = sorted(DATA_USERDATA.glob(f"{bvid}-*-summary.txt"))
    return {
        "transcript": transcripts[0] if transcripts else None,
        "summary": summaries[0] if summaries else None,
    }

def _merged_content(suffix: str) -> tuple[str, list[str]]:
    """Return merged content and list of missing BVIDs."""
    paths = find_userdata_files(str(DATA_USERDATA), suffix)
    if not paths:
        return "", []
    done_bvids = {Path(p).name.split("-")[0] for p in paths}
    missing = [b for b in _requested_bvids if b not in done_bvids]
    content = merge_files(paths)
    if missing:
        note = "# 注意：以下 BV ID 尚未完成：" + "、".join(missing) + "\n\n---\n\n"
        content = note + content
    return content, missing

# ---------------------------------------------------------------------------
# Background processing pipeline
# ---------------------------------------------------------------------------

def _processing_worker(
    bvids: list[str],
    api_key: str,
    model: str,
    base_url: str,
    summary_prompt: str,
    cookie: str,
) -> None:
    global _is_processing
    try:
        prompt = summary_prompt.strip() or DEFAULT_SUMMARY_PROMPT
        pipe = queue.Queue(maxsize=1)
        _DONE = object()

        def downloader():
            for idx, bvid in enumerate(bvids):
                if idx > 0:
                    delay = random.randint(20, 60)
                    _emit("info", f"等待 {delay} 秒…")
                    time.sleep(delay)

                ex = _check_existing(bvid)
                if ex["transcript"] and ex["summary"]:
                    _emit("info", f"{bvid}: 已完成，跳过")
                    pipe.put({"bvid": bvid, "skip_all": True})
                    continue

                if ex["transcript"]:
                    _emit("info", f"{bvid}: 文字稿已存在，仅生成摘要")
                    pipe.put({"bvid": bvid, "wav_path": None, "safe_title": None, "existing": ex})
                    continue

                try:
                    _emit("info", f"下载 {bvid}…")
                    r = download_bilibili_wav(bvid, str(DATA_TEMP), cookie or None, False)
                    ex = _check_existing(bvid)
                    pipe.put({"bvid": bvid, "wav_path": r["wav_path"], "safe_title": r["safe_title"], "existing": ex})
                except Exception as e:
                    _emit("error", f"{bvid} 下载失败: {e}")
                    pipe.put({"bvid": bvid, "error": True})

            pipe.put(_DONE)

        total = len(bvids)
        done_count = [0]  # mutable counter shared with processor closure

        def processor():
            _model_ready.wait(timeout=300)
            while True:
                item = pipe.get()
                if item is _DONE:
                    break

                bvid = item["bvid"]

                if item.get("skip_all"):
                    done_count[0] += 1
                    _emit("info", f"({done_count[0]}/{total}) {bvid}: 已完成，跳过")
                    continue
                if item.get("error"):
                    done_count[0] += 1
                    _emit("error", f"({done_count[0]}/{total}) {bvid}: 下载失败")
                    continue

                wav_path = item.get("wav_path")
                safe_title = item.get("safe_title")
                ex = item.get("existing", {}) or {}

                try:
                    # Determine stem for file naming
                    if safe_title:
                        stem = f"{bvid}-{safe_title}"
                    elif ex.get("transcript"):
                        stem = ex["transcript"].name.removesuffix("-transcript.txt")
                    else:
                        stem = bvid

                    t_path = DATA_USERDATA / f"{stem}-transcript.txt"
                    s_path = DATA_USERDATA / f"{stem}-summary.txt"

                    # Transcription
                    if not ex.get("transcript"):
                        if _recognizer is None:
                            raise RuntimeError("ASR 模型未就绪")
                        _emit("info", f"转录 {bvid}…")
                        raw = asr_transcribe(_recognizer, wav_path)
                        Path(wav_path).unlink(missing_ok=True)

                        if api_key:
                            _emit("info", f"LLM 补全 {bvid}…")
                            completed = complete_transcription(raw, bvid, api_key, model, base_url)
                        else:
                            completed = f"## BVID: {bvid}\n\n{raw}"

                        t_path.write_text(completed, encoding="utf-8")
                        _emit("success", f"{bvid} 文字稿已保存")
                    else:
                        completed = t_path.read_text(encoding="utf-8")

                    # Summarization (always if api_key, with default prompt)
                    if not ex.get("summary"):
                        if api_key:
                            _emit("info", f"生成摘要 {bvid}…")
                            summary = summarize_text(completed, bvid, prompt, api_key, model, base_url)
                            s_path.write_text(summary, encoding="utf-8")
                            _emit("success", f"{bvid} 摘要已保存")
                        else:
                            _emit("info", f"{bvid}: 无 API Key，跳过摘要")

                    done_count[0] += 1
                    _emit("success", f"({done_count[0]}/{total}) {bvid} 完成")

                except Exception as e:
                    done_count[0] += 1
                    _emit("error", f"({done_count[0]}/{total}) {bvid} 处理失败: {e}")
                    if wav_path:
                        Path(wav_path).unlink(missing_ok=True)

        dl = threading.Thread(target=downloader, daemon=True)
        pr = threading.Thread(target=processor, daemon=True)
        dl.start()
        pr.start()
        dl.join()
        pr.join()
        _emit("success", "所有任务完成")

    finally:
        with _process_lock:
            _is_processing = False

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", providers=PROVIDERS)

@app.route("/config")
def config_page():
    return render_template("config.html", providers=PROVIDERS)

# ── Config ──────────────────────────────────────────────────────────────────

CONFIG_PATH = DATA_USERDATA / "config.json"

@app.route("/api/config", methods=["GET"])
def get_config():
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return jsonify(data)
        except Exception:
            pass
    return jsonify({})

@app.route("/api/config", methods=["POST"])
def save_config():
    data = request.get_json(force=True, silent=True) or {}
    CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonify({"ok": True})

# ── Process ─────────────────────────────────────────────────────────────────

@app.route("/api/process", methods=["POST"])
def start_process():
    global _is_processing, status_messages, _requested_bvids

    with _process_lock:
        if _is_processing:
            return jsonify({"ok": False, "error": "已有任务在运行"}), 409

        body = request.get_json(force=True, silent=True) or {}
        raw = body.get("bvids", [])
        if isinstance(raw, str):
            raw = raw.splitlines()
        bvids = [b.strip() for b in raw if b.strip()]

        if not bvids:
            return jsonify({"ok": False, "error": "未提供 BV ID"}), 400

        api_key = body.get("api_key", "").strip()
        model = body.get("model", "deepseek-chat").strip()
        base_url = body.get("base_url", "https://api.deepseek.com/v1").strip()
        summary_prompt = body.get("summary_prompt", "").strip()
        cookie = body.get("cookie", "").strip()

        status_messages = []
        _requested_bvids = bvids
        _is_processing = True

    threading.Thread(
        target=_processing_worker,
        args=(bvids, api_key, model, base_url, summary_prompt, cookie),
        daemon=True,
    ).start()
    return jsonify({"ok": True, "count": len(bvids)})

# ── SSE status stream ────────────────────────────────────────────────────────

@app.route("/api/status")
def status_stream():
    def generate():
        last_sent = 0
        while True:
            with _status_lock:
                current_len = len(status_messages)
            while last_sent < current_len:
                with _status_lock:
                    msg = status_messages[last_sent]
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
                last_sent += 1
                with _status_lock:
                    current_len = len(status_messages)
            if not _is_processing and last_sent >= len(status_messages):
                time.sleep(0.5)
                if not _is_processing and last_sent >= len(status_messages):
                    break
            time.sleep(0.1)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ── File listing ─────────────────────────────────────────────────────────────

@app.route("/api/bvids")
def list_bvids():
    """Return unique BV IDs that have existing transcript or summary files."""
    paths = find_userdata_files(str(DATA_USERDATA), "-transcript.txt") + \
            find_userdata_files(str(DATA_USERDATA), "-summary.txt")
    bvids = sorted({Path(p).name.split("-")[0] for p in paths})
    return jsonify({"bvids": bvids})


@app.route("/api/files")
def list_files():
    transcripts = find_userdata_files(str(DATA_USERDATA), "-transcript.txt")
    summaries = find_userdata_files(str(DATA_USERDATA), "-summary.txt")
    files = [
        {"name": Path(p).name, "type": "transcript"} for p in transcripts
    ] + [
        {"name": Path(p).name, "type": "summary"} for p in summaries
    ]
    return jsonify({"files": files})

# ── View / Download helpers ──────────────────────────────────────────────────

@app.route("/api/view/transcript")
def view_transcript():
    content, _ = _merged_content("-transcript.txt")
    if not content:
        return "暂无文字稿", 404
    return Response(content, mimetype="text/plain; charset=utf-8")

@app.route("/api/view/summary")
def view_summary():
    content, _ = _merged_content("-summary.txt")
    if not content:
        return "暂无摘要", 404
    return Response(content, mimetype="text/plain; charset=utf-8")

@app.route("/api/download/transcript")
def download_transcript():
    content, _ = _merged_content("-transcript.txt")
    if not content:
        return jsonify({"error": "暂无文字稿"}), 404
    tmp = DATA_TEMP / "merged-transcript.txt"
    tmp.write_text(content, encoding="utf-8")
    return send_file(
        str(tmp),
        as_attachment=True,
        download_name="merged-transcript.txt",
        mimetype="text/plain",
    )

@app.route("/api/download/summary")
def download_summary():
    content, _ = _merged_content("-summary.txt")
    if not content:
        return jsonify({"error": "暂无摘要"}), 404
    tmp = DATA_TEMP / "merged-summary.txt"
    tmp.write_text(content, encoding="utf-8")
    return send_file(
        str(tmp),
        as_attachment=True,
        download_name="merged-summary.txt",
        mimetype="text/plain",
    )

# ── Clear operations ─────────────────────────────────────────────────────────

@app.route("/api/clear/temp", methods=["POST"])
def clear_temp():
    cleared = 0
    for item in DATA_TEMP.iterdir():
        try:
            if item.is_file():
                item.unlink()
                cleared += 1
            elif item.is_dir():
                shutil.rmtree(item)
                cleared += 1
        except OSError:
            pass
    return jsonify({"ok": True, "cleared": cleared})

@app.route("/api/clear/all", methods=["POST"])
def clear_all():
    cleared = 0
    for item in DATA_USERDATA.iterdir():
        if item.name == "config.json":
            continue
        try:
            if item.is_file():
                item.unlink()
                cleared += 1
            elif item.is_dir():
                shutil.rmtree(item)
                cleared += 1
        except OSError:
            pass
    return jsonify({"ok": True, "cleared": cleared})

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
