"""
a2doc Flask web application.
Bilibili audio downloader and transcriber with LLM completion and summarisation.
"""

import json
import os
import queue
import random
import shutil
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
from src.utils.asr import transcribe_audio  # noqa: E402
from src.utils.llm import complete_transcription, summarize_text, PROVIDERS  # noqa: E402
from src.utils.merge import merge_files, find_userdata_files  # noqa: E402

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Processing state (module-level singletons)
# ---------------------------------------------------------------------------
status_messages: list[dict] = []          # accumulated SSE message dicts
_status_lock = threading.Lock()
processing_lock = threading.Lock()
is_processing = False


def _emit(msg_type: str, text: str) -> None:
    """Append a status message visible to the SSE endpoint."""
    with _status_lock:
        status_messages.append({"type": msg_type, "text": text})


# ---------------------------------------------------------------------------
# Background processing thread
# ---------------------------------------------------------------------------

def _processing_worker(
    bvids: list[str],
    api_key: str,
    model: str,
    base_url: str,
    summary_prompt: str,
    cookie: str,
) -> None:
    global is_processing
    try:
        for idx, bvid in enumerate(bvids):
            # Random delay between videos (not before the first one)
            if idx > 0:
                delay = random.randint(20, 60)
                _emit("info", f"等待 {delay} 秒后处理下一个视频…")
                time.sleep(delay)

            try:
                # ── 1. Download ──────────────────────────────────────────
                _emit("info", f"正在下载 {bvid}…")
                dl_result = download_bilibili_wav(
                    bvid=bvid,
                    output_dir=str(DATA_TEMP),
                    cookie=cookie or None,
                    show_progress=False,
                )
                wav_path = dl_result["wav_path"]
                safe_title = dl_result["safe_title"]
                title = dl_result["title"]
                _emit("info", f"下载完成：{title}")

                # ── 2. Transcribe ────────────────────────────────────────
                _emit("info", f"正在转录 {bvid}…")
                raw_text = transcribe_audio(wav_path)
                _emit("info", f"转录完成，共 {len(raw_text)} 字符")

                # Remove the WAV after transcription to save space
                try:
                    Path(wav_path).unlink(missing_ok=True)
                except OSError:
                    pass

                # ── 3. LLM completion (optional) ─────────────────────────
                if api_key:
                    _emit("info", f"正在用 LLM 补全文字稿：{bvid}…")
                    completed_text = complete_transcription(
                        text=raw_text,
                        bvid=bvid,
                        api_key=api_key,
                        model=model,
                        base_url=base_url,
                    )
                else:
                    completed_text = f"## BVID: {bvid}\n\n{raw_text}"

                transcript_path = DATA_USERDATA / f"{bvid}-{safe_title}-transcript.txt"
                transcript_path.write_text(completed_text, encoding="utf-8")
                _emit("info", f"文字稿已保存：{transcript_path.name}")

                # ── 4. Summarisation (optional) ──────────────────────────
                if api_key and summary_prompt:
                    _emit("info", f"正在生成摘要：{bvid}…")
                    summary = summarize_text(
                        text=completed_text,
                        bvid=bvid,
                        summary_prompt=summary_prompt,
                        api_key=api_key,
                        model=model,
                        base_url=base_url,
                    )
                    summary_path = DATA_USERDATA / f"{bvid}-{safe_title}-summary.txt"
                    summary_path.write_text(summary, encoding="utf-8")
                    _emit("info", f"摘要已保存：{summary_path.name}")

                _emit("success", f"完成：{bvid} — {title}")

            except Exception as exc:
                _emit("error", f"处理 {bvid} 时出错：{exc}")

        _emit("success", "所有任务完成")
    finally:
        with processing_lock:
            is_processing = False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", providers=PROVIDERS)


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
    global is_processing, status_messages

    with processing_lock:
        if is_processing:
            return jsonify({"ok": False, "error": "已有任务在运行，请等待完成"}), 409

        body = request.get_json(force=True, silent=True) or {}
        raw_bvids = body.get("bvids", [])
        # Accept newline-separated string or list
        if isinstance(raw_bvids, str):
            raw_bvids = raw_bvids.splitlines()
        bvids = [b.strip() for b in raw_bvids if b.strip()]

        if not bvids:
            return jsonify({"ok": False, "error": "未提供任何 BV ID"}), 400

        api_key = body.get("api_key", "").strip()
        model = body.get("model", "deepseek-chat").strip()
        base_url = body.get("base_url", "https://api.deepseek.com/v1").strip()
        summary_prompt = body.get("summary_prompt", "").strip()
        cookie = body.get("cookie", "").strip()

        # Reset status log for this run
        status_messages = []
        is_processing = True

    thread = threading.Thread(
        target=_processing_worker,
        args=(bvids, api_key, model, base_url, summary_prompt, cookie),
        daemon=True,
    )
    thread.start()
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
            # If processing is done and we have sent everything, close the stream
            if not is_processing and last_sent >= len(status_messages):
                time.sleep(0.5)
                if not is_processing and last_sent >= len(status_messages):
                    break
            time.sleep(0.1)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── File listing ─────────────────────────────────────────────────────────────

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

def _merged_content(suffix: str) -> str:
    paths = find_userdata_files(str(DATA_USERDATA), suffix)
    if not paths:
        return ""
    return merge_files(paths)


@app.route("/api/view/transcript")
def view_transcript():
    content = _merged_content("-transcript.txt")
    if not content:
        return "暂无文字稿", 404
    return Response(content, mimetype="text/plain; charset=utf-8")


@app.route("/api/view/summary")
def view_summary():
    content = _merged_content("-summary.txt")
    if not content:
        return "暂无摘要", 404
    return Response(content, mimetype="text/plain; charset=utf-8")


@app.route("/api/download/transcript")
def download_transcript():
    content = _merged_content("-transcript.txt")
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
    content = _merged_content("-summary.txt")
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
        # Preserve config.json
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
