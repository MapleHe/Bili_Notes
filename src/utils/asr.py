#!/usr/bin/env python3
"""
中文语音转文字脚本
用法: python asr.py <input_audio> [output.txt] [--threads N]
依赖: pip install sherpa-onnx soundfile numpy
      pkg install ffmpeg (Termux) 或 apt install ffmpeg (Linux/macOS)
"""

import sys
import os
import math
import platform
import subprocess
import argparse
import urllib.request
import tarfile
from pathlib import Path

import soundfile as sf
import numpy as np
import sherpa_onnx

# ── 路径配置（相对于脚本位置自动计算）────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # src/utils/ -> src/ -> project root
MODELS_DIR = PROJECT_ROOT / "models"
SENSE_VOICE_DIR = MODELS_DIR / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09"
PARAFORMER_DIR = MODELS_DIR / "sherpa-onnx-paraformer-zh-int8-2025-10-07"
FIRE_RED_DIR = MODELS_DIR / "sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25"

# ── 运行配置 ──────────────────────────────────────────────────────
MODEL_TYPE = "sense_voice"  # 可选: "sense_voice" / "paraformer" / "fire_red"


def _detect_num_threads() -> int:
    """Auto-detect optimal thread count based on platform and resource limits.

    Goal: use at most 1/4 of CPU resources per Goals.v1.0 constraint on M2.
    On Linux (ThinkPad/Termux): use 1/4 of cores, capped at 4.
    """
    total_cores = os.cpu_count() or 4
    system = platform.system()
    machine = platform.machine()

    if system == "Darwin" and machine == "arm64":
        # Apple Silicon M-series: 1/4 of total cores, minimum 2
        return max(2, total_cores // 4)
    elif system == "Linux":
        # ThinkPad or Termux: 1/4 of cores, clamped to [2, 4]
        return max(2, min(4, total_cores // 4))
    else:
        return 2


NUM_THREADS = _detect_num_threads()

# ── 模型下载地址 ──────────────────────────────────────────────────
_MODEL_URLS = {
    "sense_voice": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
        "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09.tar.bz2"
    ),
}

_MODEL_REQUIRED_FILES = {
    "sense_voice": ["model.int8.onnx", "tokens.txt"],
}

_MODEL_DIRS = {
    "sense_voice": SENSE_VOICE_DIR,
}
# ─────────────────────────────────────────────────────────────────


def ensure_model(model_type: str = "sense_voice") -> None:
    """Check if the model exists; if not, download and extract it automatically."""
    if model_type not in _MODEL_DIRS:
        raise ValueError(f"未知模型类型: {model_type}，支持的类型: {list(_MODEL_DIRS.keys())}")

    model_dir = _MODEL_DIRS[model_type]
    required_files = _MODEL_REQUIRED_FILES.get(model_type, [])

    if model_dir.exists() and all((model_dir / f).exists() for f in required_files):
        return

    if model_type not in _MODEL_URLS:
        raise RuntimeError(
            f"模型目录不存在: {model_dir}\n"
            f"请手动下载并解压到 {MODELS_DIR}"
        )

    url = _MODEL_URLS[model_type]
    archive_name = url.split("/")[-1]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = MODELS_DIR / archive_name

    print(f"模型未找到，开始下载: {archive_name}")
    print(f"下载地址: {url}")
    _download_with_progress(url, archive_path)

    print(f"\n解压中: {archive_name} -> {MODELS_DIR}")
    with tarfile.open(archive_path, "r:bz2") as tar:
        tar.extractall(path=MODELS_DIR)
    print("解压完成。")

    archive_path.unlink(missing_ok=True)

    if not model_dir.exists() or not all((model_dir / f).exists() for f in required_files):
        raise RuntimeError(f"解压后未找到预期的模型文件，请检查 {model_dir}")
    print(f"模型已就绪: {model_dir}")


def _download_with_progress(url: str, dest: Path) -> None:
    """Download a file from url to dest, showing a progress bar."""
    def _report(block_count, block_size, total_size):
        downloaded = block_count * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            done = int(pct / 2)
            bar = "#" * done + "-" * (50 - done)
            sys.stdout.write(
                f"\r  [{bar}] {pct:5.1f}%  "
                f"{downloaded // 1024 // 1024}MB / {total_size // 1024 // 1024}MB"
            )
        else:
            sys.stdout.write(f"\r  已下载 {downloaded // 1024 // 1024}MB")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=_report)


def _probe_duration(input_path: str) -> float | None:
    """Return audio duration in seconds via ffprobe, or None on failure."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return float(out.stdout.strip())
    except Exception:
        return None


def _use_coreml() -> bool:
    """Return True if running on Apple Silicon where CoreML may accelerate inference."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _build_recognizer_kwargs(model_type: str, provider: str) -> dict:
    """Build keyword arguments for the recognizer constructor."""
    if model_type == "sense_voice":
        return dict(
            model=str(SENSE_VOICE_DIR / "model.int8.onnx"),
            tokens=str(SENSE_VOICE_DIR / "tokens.txt"),
            num_threads=NUM_THREADS,
            use_itn=True,
            language="zh",
            debug=False,
            provider=provider,
        )
    elif model_type == "paraformer":
        return dict(
            paraformer=str(PARAFORMER_DIR / "model.int8.onnx"),
            tokens=str(PARAFORMER_DIR / "tokens.txt"),
            num_threads=NUM_THREADS,
            debug=False,
            provider=provider,
        )
    elif model_type == "fire_red":
        return dict(
            model_dir=str(FIRE_RED_DIR),
            num_threads=NUM_THREADS,
            debug=False,
            provider=provider,
        )
    else:
        raise ValueError(f"未知模型类型: {model_type}")


def _create_recognizer(model_type: str, **kwargs) -> sherpa_onnx.OfflineRecognizer:
    if model_type == "sense_voice":
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(**kwargs)
    elif model_type == "paraformer":
        return sherpa_onnx.OfflineRecognizer.from_paraformer(**kwargs)
    elif model_type == "fire_red":
        return sherpa_onnx.OfflineRecognizer.from_fire_red_asr(**kwargs)
    else:
        raise ValueError(f"未知模型类型: {model_type}")


def load_recognizer(model_type: str = MODEL_TYPE) -> sherpa_onnx.OfflineRecognizer:
    """Load the recognizer for the given model type.

    On Apple Silicon, attempts CoreML acceleration first and falls back to CPU
    if the installed sherpa-onnx version does not support the provider parameter.
    """
    provider = "coreml" if _use_coreml() else "cpu"
    kwargs = _build_recognizer_kwargs(model_type, provider)

    try:
        return _create_recognizer(model_type, **kwargs)
    except TypeError:
        if provider == "coreml":
            # Installed sherpa-onnx does not support provider= parameter — fall back to CPU
            print(
                "[警告] 当前 sherpa-onnx 版本不支持 CoreML，已回退到 CPU 模式。"
                " 如需 CoreML 加速，请使用支持 CoreML 的 sherpa-onnx 版本。",
                file=sys.stderr,
            )
            kwargs.pop("provider", None)
            return _create_recognizer(model_type, **kwargs)
        raise


def transcribe_chunked(
    recognizer: sherpa_onnx.OfflineRecognizer,
    wav_path: str,
    chunk_duration_sec: int = 30,
    on_progress=None,
) -> str:
    """Read a 16kHz mono WAV file in chunks and transcribe each chunk.

    Peak memory usage: ~model size + one chunk buffer (~960 KB for 30s at 16kHz).
    on_progress: optional callable(chunks_done: int, total_chunks: int)
    """
    texts = []
    with sf.SoundFile(wav_path) as f:
        if f.samplerate != 16000:
            raise ValueError(f"采样率应为 16000，实际为 {f.samplerate}，请重新转码")
        chunk_size = chunk_duration_sec * f.samplerate
        total_chunks = max(1, math.ceil(len(f) / chunk_size))
        chunks_done = 0
        while True:
            data = f.read(chunk_size, dtype="float32")
            if len(data) == 0:
                break
            stream = recognizer.create_stream()
            stream.accept_waveform(f.samplerate, data)
            recognizer.decode_stream(stream)
            text = stream.result.text.strip()
            if text:
                texts.append(text)
            chunks_done += 1
            if on_progress:
                on_progress(chunks_done, total_chunks)
    return " ".join(texts)


def transcribe_from_audio(
    recognizer: sherpa_onnx.OfflineRecognizer,
    input_path: str,
    chunk_duration_sec: int = 30,
    on_progress=None,
) -> str:
    """Convert audio to PCM via ffmpeg pipe and transcribe in chunks.

    Accepts any ffmpeg-compatible audio format (WAV, M4A, MP3, etc.).
    No temporary file is written to disk — ffmpeg output is piped directly to ASR.
    Peak memory: ~model size + one chunk buffer (~960 KB for 30s at 16kHz).
    on_progress: optional callable(chunks_done: int, total_chunks: int | None)
    """
    sample_rate = 16000
    chunk_samples = chunk_duration_sec * sample_rate
    bytes_per_sample = 2  # pcm_s16le = 16-bit signed little-endian
    chunk_bytes = chunk_samples * bytes_per_sample

    duration = _probe_duration(input_path)
    total_chunks = math.ceil(duration / chunk_duration_sec) if duration else None

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", str(sample_rate),
        "-ac", "1",
        "-f", "s16le",
        "-c:a", "pcm_s16le",
        "pipe:1",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    texts = []
    chunks_done = 0

    try:
        while True:
            raw = proc.stdout.read(chunk_bytes)
            if not raw:
                break
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples)
            recognizer.decode_stream(stream)
            text = stream.result.text.strip()
            if text:
                texts.append(text)
            chunks_done += 1
            if on_progress:
                on_progress(chunks_done, total_chunks)
    finally:
        proc.stdout.close()
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 转码失败 (返回码 {proc.returncode})，请检查输入文件: {input_path}")

    return " ".join(texts)


def transcribe(recognizer: sherpa_onnx.OfflineRecognizer, wav_path: str, on_progress=None) -> str:
    """Read a WAV file and return the transcribed text (chunked to limit memory)."""
    return transcribe_chunked(recognizer, wav_path, on_progress=on_progress)


def transcribe_audio(input_path: str) -> str:
    """Load the recognizer and transcribe an audio file, returning the text string.

    Accepts any ffmpeg-compatible audio format. Uses ffmpeg pipe streaming to avoid
    writing a temporary WAV file. Ensures the model is present before loading.
    The caller is responsible for any file writing.
    """
    ensure_model(MODEL_TYPE)
    recognizer = load_recognizer(MODEL_TYPE)
    return transcribe_from_audio(recognizer, input_path)


def transcribe_to_file(input_path: str, output_path: str) -> str:
    """Transcribe an audio file and write the result to output_path.

    Returns the transcribed text string.
    """
    text = transcribe_audio(input_path)
    Path(output_path).write_text(text + "\n", encoding="utf-8")
    return text


def main():
    global NUM_THREADS
    parser = argparse.ArgumentParser(description="中文语音转文字（sherpa-onnx）")
    parser.add_argument("input", help="输入音频文件（WAV/M4A/MP3 等，自动转码）")
    parser.add_argument(
        "output",
        nargs="?",
        help="输出 txt 文件路径（可选，默认与输入同名 .txt）",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        metavar="N",
        help=f"线程数（默认自动检测: {NUM_THREADS}）",
    )
    args = parser.parse_args()

    if args.threads is not None:
        NUM_THREADS = args.threads

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误：找不到文件 {args.input}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output if args.output else str(input_path.with_suffix(".txt"))

    coreml_label = "是" if _use_coreml() else "否"
    print(f"[1/3] 检查模型（{MODEL_TYPE}）...")
    ensure_model(MODEL_TYPE)

    print(f"[2/3] 加载模型（{MODEL_TYPE}，线程数={NUM_THREADS}，CoreML={coreml_label}）...")
    recognizer = load_recognizer(MODEL_TYPE)

    print("[3/3] 识别中（ffmpeg 管道 + 分块处理，无临时文件）...")
    text = transcribe_from_audio(recognizer, str(input_path))

    Path(output_path).write_text(text + "\n", encoding="utf-8")
    print(f"结果已写入：{output_path}")


if __name__ == "__main__":
    main()
