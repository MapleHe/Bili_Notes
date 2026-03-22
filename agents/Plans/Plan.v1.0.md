# Plan v1.0 — MacBook Air M2 Performance Optimization

**Goal**: Optimize Bili_Notes for MacBook Air M2 (16GB RAM, 256GB storage), using at most 1/4 of resources (4GB RAM, ~2 CPU cores). Secondary goal: improve performance on Lenovo ThinkPad T14 Gen 2i (8 CPU, 32GB, no GPU).

**Date**: 2026-03-21

---

## Problem Analysis

The current `asr.py` uses SenseVoice (Int8, ~300MB model) with `OfflineRecognizer`, which:
1. Loads the **entire WAV into memory** via `sf.read()` — problematic for long audio (a 2-hour video at 16kHz mono 16-bit = ~230MB WAV)
2. Processes the whole file in **one blocking call** — no streaming, no chunking
3. Uses **4 threads** hardcoded — no platform detection
4. Has **no CoreML/GPU acceleration** on macOS — runs CPU-only
5. ffmpeg conversion creates a **full temporary WAV file** on disk before ASR starts

---

## Plan Overview

| Phase | Description | Priority | Status |
|-------|-------------|----------|--------|
| **2** | Platform-aware configuration (M2 vs x86) | High | TODO |
| **1A** | Chunked offline ASR (SenseVoice, 30s segments) | High | TODO |
| **3** | ffmpeg pipe streaming (skip temp WAV) | High | TODO |
| **4** | CoreML acceleration on macOS (CPU fallback) | High | TODO |
| **5** | VAD integration for silence skipping | Deferred | Skip unless benchmarks show need |
| **1B** | Alternative model evaluation (Zipformer/Paraformer) | Deferred | Skip unless chunked SenseVoice is still too slow |

---

## Phase 1A: Chunked Offline ASR (SenseVoice)

**Problem**: `sf.read()` loads full audio into memory; `OfflineRecognizer` processes it all at once. For a 2-hour video, this uses ~2.5GB+ RAM. SenseVoice uses attention, so processing time grows quadratically with sequence length — this is the root cause of slowness on the ThinkPad for long audio.

**Decision (2026-03-21)**: Keep SenseVoice (accuracy confirmed acceptable by administrator). Chunk the audio manually into 30-second segments, process each independently, concatenate text. This solves both memory and speed problems without changing models. Zipformer/Paraformer evaluation deferred — revisit only if chunked SenseVoice is still too slow after benchmarking.

### Implementation (Phase 1A — Chunked Offline, simplest)

In `asr.py`:

```python
def transcribe_chunked(recognizer, wav_path, chunk_duration_sec=30):
    """Read WAV in chunks and transcribe each chunk."""
    import soundfile as sf

    texts = []
    with sf.SoundFile(wav_path) as f:
        sample_rate = f.samplerate
        chunk_size = chunk_duration_sec * sample_rate
        while True:
            data = f.read(chunk_size, dtype="float32")
            if len(data) == 0:
                break
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, data)
            recognizer.decode_stream(stream)
            text = stream.result.text.strip()
            if text:
                texts.append(text)
    return " ".join(texts)
```

**Benefits**:
- Peak memory: ~model size + one chunk buffer (~1MB for 30s at 16kHz)
- No model change needed, keeps SenseVoice accuracy
- Works on all platforms

### Implementation (Phase 1B — Online Streaming, optional future)

Add Zipformer-Transducer online model support:

```python
# New model entry in _MODEL_URLS / _MODEL_DIRS
"zipformer_online": {
    "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/...",
    "dir": MODELS_DIR / "sherpa-onnx-streaming-zipformer-...",
}

def load_online_recognizer(model_type):
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=str(model_dir / "encoder-epoch-99-avg-1.int8.onnx"),
        decoder=str(model_dir / "decoder-epoch-99-avg-1.int8.onnx"),
        joiner=str(model_dir / "joiner-epoch-99-avg-1.int8.onnx"),
        tokens=str(model_dir / "tokens.txt"),
        num_threads=NUM_THREADS,
    )
```

### Files to modify
- `src/utils/asr.py` — add `transcribe_chunked()`, update `transcribe()` to use it for long audio

---

## Phase 2: Platform-Aware Configuration

**Problem**: `NUM_THREADS = 4` is hardcoded for Snapdragon 8 Gen 1. On M2 with 1/4 resource limit, we should use 2 threads. On ThinkPad with 8 CPU cores, 2-4 threads.

### Implementation

In `asr.py`, replace the hardcoded `NUM_THREADS`:

```python
import platform
import os

def _detect_num_threads() -> int:
    """Auto-detect optimal thread count based on platform and resource limits."""
    total_cores = os.cpu_count() or 4
    system = platform.system()
    machine = platform.machine()

    if system == "Darwin" and machine == "arm64":
        # Apple Silicon: use 1/4 of cores (per Goals.v1.0 constraint)
        return max(2, total_cores // 4)
    elif system == "Linux":
        # ThinkPad or Termux: use 1/4 to 1/2 of cores
        return max(2, min(4, total_cores // 4))
    else:
        return 2

NUM_THREADS = _detect_num_threads()
```

Also add a `--threads` CLI argument to override.

### Files to modify
- `src/utils/asr.py` — replace `NUM_THREADS` constant with `_detect_num_threads()`

---

## Phase 3: FFmpeg Pipe Streaming (Eliminate Temp WAV)

**Problem**: Current flow creates a full temporary WAV file on disk, then reads it. For a 2-hour video, this is ~230MB of disk I/O.

**Solution**: Pipe ffmpeg output directly to ASR via stdout, combined with Phase 1's chunked reading.

### Implementation

```python
def transcribe_from_audio(recognizer, input_path, chunk_duration_sec=30):
    """Convert audio to WAV via ffmpeg pipe and transcribe in chunks."""
    sample_rate = 16000
    chunk_samples = chunk_duration_sec * sample_rate
    bytes_per_sample = 2  # pcm_s16le
    chunk_bytes = chunk_samples * bytes_per_sample

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", str(sample_rate),
        "-ac", "1",
        "-f", "s16le",      # raw PCM, no WAV header
        "-c:a", "pcm_s16le",
        "pipe:1",            # output to stdout
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    texts = []

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

    proc.wait()
    return " ".join(texts)
```

**Benefits**:
- Zero temp WAV file on disk
- Streaming from ffmpeg → ASR
- Memory: only one chunk in memory at a time (~960KB for 30s)
- Works for `download_bilibili_wav()` too — can pipe directly from download

### macOS-specific ffmpeg optimization

On macOS, use AudioToolbox for faster decoding:

```python
def _get_ffmpeg_input_opts(input_path):
    """Platform-specific ffmpeg input options."""
    opts = ["-i", input_path]
    # AudioToolbox is used automatically by ffmpeg on macOS for AAC/ALAC
    # No special flags needed — just ensure ffmpeg is built with --enable-audiotoolbox
    return opts
```

Note: For audio-only decoding, the speedup from AudioToolbox is marginal since audio decoding is not the bottleneck. The real gain is from eliminating the temp file.

### Files to modify
- `src/utils/asr.py` — add `transcribe_from_audio()`, update `main()` and `transcribe_audio()`
- `src/utils/extract_url.py` — update `download_bilibili_wav()` to optionally skip WAV conversion (pipe directly)

---

## Phase 4: CoreML Acceleration on macOS

**Problem**: sherpa-onnx uses ONNX Runtime's CPU execution provider by default. On M2, CoreML can dispatch operations to GPU via Metal.

**Caveat**: CoreML acceleration for ASR models is not always faster due to model compilation overhead and operator support gaps. This should be **tested empirically**.

### Implementation

```python
def _get_provider_kwargs():
    """Get platform-specific ONNX Runtime provider settings."""
    system = platform.system()
    machine = platform.machine()

    if system == "Darwin" and machine == "arm64":
        return {
            "provider": "coreml",
            "provider_config": "CPUAndGPU",  # avoid ANE for ASR
        }
    return {}
```

**Important**: sherpa-onnx's Python API may not expose CoreML provider configuration directly. Need to verify:
1. Check if `sherpa_onnx.OfflineRecognizer` accepts `provider` parameter
2. If not, may need to build sherpa-onnx from source with CoreML enabled
3. Alternative: use `onnxruntime` directly with CoreML EP, bypassing sherpa-onnx wrapper

### Risk
- CoreML model compilation on first run can take 30-60 seconds
- Some ONNX ops may not be supported by CoreML, causing fallback to CPU
- May not provide meaningful speedup for Int8 models (CoreML prefers Float16)

### Decision (2026-03-21)
- **Confirmed by administrator**: Pursue CoreML on M2, with CPU-only fallback on devices without GPU (ThinkPad, Termux)
- Implement after Phases 1A-3, but do not defer indefinitely — this is a required deliverable

### Files to modify
- `src/utils/asr.py` — add provider detection in `load_recognizer()`

---

## Phase 5: VAD Integration (Voice Activity Detection)

**Problem**: Long audio may contain significant silence (pauses, intros, outros). Processing silence wastes CPU time.

**Solution**: sherpa-onnx includes a built-in VAD (Silero VAD) that can segment audio into speech regions.

### Implementation sketch

```python
import sherpa_onnx

def create_vad():
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(MODELS_DIR / "silero_vad.onnx")
    config.silero_vad.threshold = 0.5
    config.silero_vad.min_silence_duration = 0.5
    config.silero_vad.min_speech_duration = 0.25
    config.sample_rate = 16000
    return sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)

def transcribe_with_vad(recognizer, wav_path):
    vad = create_vad()
    # Feed audio chunks to VAD, only transcribe speech segments
    ...
```

**Benefits**:
- Skip silence → fewer inference calls → faster total time
- Better text output (no garbled noise transcription)
- Silero VAD model is tiny (~2MB)

### Risk
- Adds complexity
- VAD threshold tuning needed
- Marginal benefit if audio is mostly speech

### Recommendation
- Implement after Phases 1-3, only if benchmarks show silence is a significant portion of processed audio

### Files to modify
- `src/utils/asr.py` — add VAD functions, integrate with chunked transcription

---

## Implementation Order

```
Phase 2 (platform config)     — smallest change, immediate benefit
Phase 1A (chunked offline)    — biggest memory + speed improvement
Phase 3 (ffmpeg pipe)         — eliminates temp files, completes streaming pipeline
  └─ BENCHMARK HERE: measure speed on ThinkPad with 10/30/60 min audio
Phase 4 (CoreML)              — M2 GPU acceleration, CPU fallback elsewhere
  └─ BENCHMARK HERE: compare CoreML vs CPU-only on M2
Phase 5 (VAD)                 — deferred, only if benchmarks show silence is a bottleneck
Phase 1B (alt models)         — deferred, only if chunked SenseVoice is still too slow
```

---

## Expected Resource Usage After Optimization

### MacBook Air M2 (target: ≤ 1/4 resources)

| Resource | Before | After (Phase 1-3) | Limit |
|----------|--------|-------------------|-------|
| RAM | 2.5GB+ (long audio) | ~500MB peak | 4GB |
| CPU threads | 4 | 2 | 2 cores |
| Disk temp | 230MB WAV (2hr audio) | 0 (pipe) | — |
| Model size | 300MB (SenseVoice Int8) | 300MB (same) | — |

### Lenovo ThinkPad T14 Gen 2i

| Resource | Before | After |
|----------|--------|-------|
| RAM | 2.5GB+ | ~500MB peak |
| CPU threads | 4 | 2-4 (auto-detected) |
| Disk temp | 230MB | 0 |

---

## Compatibility Notes

- All changes are backward-compatible with Termux (Android)
- No new dependencies required for Phases 1-3
- Phase 4 (CoreML) may require building sherpa-onnx with CoreML support
- Phase 5 (VAD) requires downloading Silero VAD model (~2MB)
- ffmpeg pipe approach works on all platforms (macOS, Linux, Termux)
- The Flask server (`app.py`) calls `transcribe_audio()` and `transcribe_to_file()` — these APIs remain unchanged; the optimization is internal

---

## Testing Plan

1. **Unit test**: Chunked transcription produces same text as full-file transcription (on a short test file)
2. **Memory benchmark**: Use `tracemalloc` or `/usr/bin/time -v` to measure peak RSS before/after
3. **Speed benchmark**: Compare wall-clock time for 10-min, 30-min, and 60-min audio files
4. **Platform test**: Verify auto-thread detection on macOS M2, Linux x86, and Termux ARM
5. **Pipe test**: Verify ffmpeg pipe produces identical output to temp-file approach

---

## Administrator Decisions (2026-03-21)

| # | Question | Decision |
|---|----------|----------|
| 1 | Model preference: evaluate Zipformer/Paraformer? | SenseVoice accuracy is fine. Keep it. Evaluate alternatives only if chunked SenseVoice is still too slow after benchmarking. |
| 2 | CoreML priority? | **Yes, pursue CoreML** on M2. CPU-only fallback on devices without GPU (ThinkPad, Termux). |
| 3 | VAD / silence skipping? | Not needed now. Only the final transcription file matters, not word-by-word output. Defer VAD unless benchmarks show silence is a bottleneck. |
| 4 | macOS ffmpeg? | Confirmed: ffmpeg installed via Homebrew (includes AudioToolbox by default). |
