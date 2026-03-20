#!/usr/bin/env python3
"""
中文语音转文字脚本
用法: python asr.py <input.wav> [output.txt]
依赖: pip install sherpa-onnx soundfile numpy
      pkg install ffmpeg (Termux) 或 apt install ffmpeg (Linux)
"""

import sys
import os
import subprocess
import tempfile
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
NUM_THREADS = 2             # 手机端建议 2，高端机可改 4

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

    # Check if all required files exist
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

    # Remove archive to save space
    archive_path.unlink(missing_ok=True)

    if not model_dir.exists() or not all((model_dir / f).exists() for f in required_files):
        raise RuntimeError(
            f"解压后未找到预期的模型文件，请检查 {model_dir}"
        )
    print(f"模型已就绪: {model_dir}")


def _download_with_progress(url: str, dest: Path) -> None:
    """Download a file from url to dest, showing a progress bar."""
    def _report(block_count, block_size, total_size):
        downloaded = block_count * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            done = int(pct / 2)
            bar = "#" * done + "-" * (50 - done)
            sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  {downloaded // 1024 // 1024}MB / {total_size // 1024 // 1024}MB")
        else:
            sys.stdout.write(f"\r  已下载 {downloaded // 1024 // 1024}MB")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=_report)


def convert_to_wav(input_path: str) -> str:
    """将任意音频格式转为 16kHz mono WAV，返回临时文件路径"""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        tmp.name,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 转码失败:\n{result.stderr.decode()}")
    return tmp.name


def load_recognizer(model_type: str = MODEL_TYPE) -> sherpa_onnx.OfflineRecognizer:
    """根据 model_type 加载对应 recognizer"""
    if model_type == "sense_voice":
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(SENSE_VOICE_DIR / "model.int8.onnx"),
            tokens=str(SENSE_VOICE_DIR / "tokens.txt"),
            num_threads=NUM_THREADS,
            use_itn=True,
            language="zh",
            debug=False,
        )
    elif model_type == "paraformer":
        return sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=str(PARAFORMER_DIR / "model.int8.onnx"),
            tokens=str(PARAFORMER_DIR / "tokens.txt"),
            num_threads=NUM_THREADS,
            debug=False,
        )
    elif model_type == "fire_red":
        return sherpa_onnx.OfflineRecognizer.from_fire_red_asr(
            model_dir=str(FIRE_RED_DIR),
            num_threads=NUM_THREADS,
            debug=False,
        )
    else:
        raise ValueError(f"未知模型类型: {model_type}")


def transcribe(recognizer: sherpa_onnx.OfflineRecognizer, wav_path: str) -> str:
    """读取 WAV 文件并返回识别文本"""
    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
    if sample_rate != 16000:
        raise ValueError(f"采样率应为 16000，实际为 {sample_rate}，请重新转码")
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


def transcribe_audio(wav_path: str) -> str:
    """Load the recognizer and transcribe a WAV file, returning the text string.

    Ensures the model is present (downloading if necessary) before loading.
    The caller is responsible for any file writing.
    """
    ensure_model(MODEL_TYPE)
    recognizer = load_recognizer(MODEL_TYPE)
    return transcribe(recognizer, wav_path)


def transcribe_to_file(wav_path: str, output_path: str) -> str:
    """Transcribe a WAV file and write the result to output_path.

    Returns the transcribed text string.
    """
    text = transcribe_audio(wav_path)
    Path(output_path).write_text(text + "\n", encoding="utf-8")
    return text


def main():
    parser = argparse.ArgumentParser(description="中文语音转文字（sherpa-onnx）")
    parser.add_argument("input", help="输入 WAV 文件（16kHz mono；其他格式会自动转换）")
    parser.add_argument(
        "output",
        nargs="?",
        help="输出 txt 文件路径（可选，默认与输入同名 .txt）",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误：找不到文件 {args.input}", file=sys.stderr)
        sys.exit(1)

    # Determine output path: default to same base name with .txt extension
    if args.output:
        output_path = args.output
    else:
        output_path = str(input_path.with_suffix(".txt"))

    # Convert to WAV if needed
    tmp_wav = None
    if input_path.suffix.lower() != ".wav":
        print(f"[1/4] 转码 {input_path.suffix} → WAV (16kHz mono)...")
        tmp_wav = convert_to_wav(str(input_path))
        wav_path = tmp_wav
    else:
        wav_path = str(input_path)

    try:
        print(f"[2/4] 检查模型（{MODEL_TYPE}）...")
        ensure_model(MODEL_TYPE)

        print(f"[3/4] 加载模型（{MODEL_TYPE}）...")
        recognizer = load_recognizer(MODEL_TYPE)

        print("[4/4] 识别中...")
        text = transcribe(recognizer, wav_path)

        Path(output_path).write_text(text + "\n", encoding="utf-8")
        print(f"结果已写入：{output_path}")
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.unlink(tmp_wav)


if __name__ == "__main__":
    main()
