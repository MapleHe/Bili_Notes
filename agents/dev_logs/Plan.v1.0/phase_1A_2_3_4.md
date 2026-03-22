# Plan v1.0 — Phase Log: Phases 2, 1A, 3, 4

**Date**: 2026-03-21
**File modified**: `src/utils/asr.py`

---

## Phase 2: Platform-Aware Thread Detection ✅

**Change**: Replaced `NUM_THREADS = 4` (hardcoded for Snapdragon 8 Gen 1) with `_detect_num_threads()`.

```python
def _detect_num_threads() -> int:
    total_cores = os.cpu_count() or 4
    system = platform.system()
    machine = platform.machine()
    if system == "Darwin" and machine == "arm64":
        return max(2, total_cores // 4)   # M2: 10 cores → 2 threads
    elif system == "Linux":
        return max(2, min(4, total_cores // 4))  # ThinkPad 8-core → 2 threads
    else:
        return 2
```

**Also added**: `--threads N` CLI flag to override auto-detection.

---

## Phase 1A: Chunked Offline ASR ✅

**Change**: Added `transcribe_chunked()`. The old `transcribe()` now delegates to it.

- Reads WAV in 30-second chunks via `sf.SoundFile` (streaming read, not full load)
- Each chunk is processed independently by `OfflineRecognizer`
- Peak memory: model size + ~960 KB per chunk (vs. model size + full audio file before)
- For a 2-hour file at 16kHz mono: drops from ~230 MB audio buffer to ~960 KB

---

## Phase 3: FFmpeg Pipe Streaming ✅

**Change**: Added `transcribe_from_audio()`. Updated `transcribe_audio()` and `main()` to use it.

- Spawns ffmpeg with `pipe:1` (stdout) — no temporary WAV file on disk
- Reads raw PCM (s16le) from ffmpeg stdout in 30-second chunks
- Converts each chunk: `int16 → float32 / 32768.0`
- Accepts any ffmpeg-compatible format (WAV, M4A, MP3, etc.)
- `transcribe_audio()` signature unchanged — callers (Flask server) unaffected

**Disk savings**: A 2-hour video at 16kHz mono previously created a ~230 MB temporary WAV. Now: 0 bytes temp disk usage.

---

## Phase 4: CoreML Acceleration (macOS M2) ✅

**Change**: Added `_use_coreml()` detection, `_build_recognizer_kwargs()` helper, and graceful fallback in `load_recognizer()`.

- On `Darwin/arm64`: passes `provider="coreml"` to `OfflineRecognizer.from_*`
- On all other platforms: uses `provider="cpu"`
- If the installed sherpa-onnx version does not support the `provider=` parameter (older versions), catches `TypeError`, warns the user, and retries without `provider` (CPU fallback)

**Risk acknowledged** (from Plan.v1.0): CoreML first-run compilation takes 30-60s; some ONNX ops may fall back to CPU; Int8 models may see less speedup than Float16.

---

## Key API Decisions

| Decision | Choice |
|----------|--------|
| `transcribe_audio()` signature | Unchanged — accepts `input_path` (any format) |
| `transcribe_to_file()` signature | Unchanged — first arg renamed from `wav_path` to `input_path` |
| `transcribe()` (internal) | Still accepts WAV path, now calls `transcribe_chunked()` |
| Backward compatibility | Flask server `app.py` calling these functions: unaffected |
| Removed | `convert_to_wav()` and `tempfile` import — no longer needed |

---

## Phase 5 (VAD) / Phase 1B (alt models)

Both deferred per Plan.v1.0 decision. Benchmark data needed before proceeding.
