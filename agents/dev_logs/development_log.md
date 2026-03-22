# Bili_Notes Development Log

## Project Overview
Building a Flask-based web application for Bilibili audio download and transcription.

## Agent Roles
- **Orchestrator**: Plans and coordinates all agent tasks
- **extract_url agent**: Upgrades Bilibili audio extraction with curl_cffi
- **asr agent**: Upgrades speech recognition with auto model download
- **llm agent**: Creates LLM text completion and summarization module
- **infrastructure agent**: Creates project structure, Flask server, and frontend
- **general coding agent**: Writes individual utility scripts

## Technology Stack

### Backend
| Component | Technology | Notes |
|-----------|-----------|-------|
| Web framework | Flask | Python, lightweight |
| HTTP client | curl_cffi | Browser impersonation, anti-anti-crawl |
| Speech recognition | sherpa-onnx SenseVoice | Int8 quantized, multi-language |
| LLM API | OpenAI-compatible (DeepSeek default) | Text completion & summarization |
| Audio processing | ffmpeg (system) | Format conversion |

### Frontend
| Component | Technology |
|-----------|-----------|
| UI | Plain HTML + CSS + vanilla JS |
| Real-time updates | SSE (Server-Sent Events) |

## Key Implementation Decisions

### Anti-Anti-Crawling
- **Problem**: Bilibili blocks automated requests
- **Solution**: curl_cffi with Chrome browser impersonation
- **Review needed**: May need cookie support for higher quality audio

### Sequential Processing with Random Delay
- **Problem**: Multiple BV IDs must not be processed in parallel
- **Solution**: Background thread with queue, 20-60s random delay between items
- **Risk**: Long processing times for large batches

### Model Auto-Download
- **Problem**: Model files are large (~100MB), not in repo
- **Solution**: Auto-download from sherpa-onnx GitHub releases on first use
- **Review needed**: Termux internet access and storage permissions

### File Naming
- Pattern: `<BVID>-<safe_title>-transcript.txt` / `<BVID>-<safe_title>-summary.txt`
- Safe title: alphanumeric and Chinese characters only, no spaces/punctuation

## Difficulties and Solutions
- Flask SSE for real-time progress: Use `text/event-stream` response with generator
- Termux compatibility: Use subprocess list form, pathlib for paths, avoid shell features
- Memory efficiency: Stream download to disk, process in chunks

---

## 2026-03-21 — Plan v1.0: MacBook Air M2 Performance Optimization

### Context
Administrator wants to deploy on MacBook Air M2 (16GB, 256GB) with at most 1/4 resource usage (4GB RAM, ~2 cores). Current SenseVoice ASR is extremely slow on long audio files on both Termux and ThinkPad T14.

### Root Cause Analysis
SenseVoice uses attention — processing time grows **quadratically** with audio length. The code loads the entire WAV into memory (`sf.read()`) and processes it in one `decode_stream()` call. A 2-hour video uses ~2.5GB+ RAM and takes very long to process.

### Plan Created
See `agents/Plans/Plan.v1.0.md` for full details. Summary:
- Phase 2: Platform-aware thread count (auto-detect cores, limit to 1/4)
- Phase 1A: Chunk audio into 30s segments, process each independently
- Phase 3: Pipe ffmpeg output directly to ASR (no temp WAV file)
- Phase 4: CoreML GPU acceleration on M2 (CPU fallback on ThinkPad/Termux)
- Phase 5/1B: Deferred (VAD, alternative models)

### Questions & Administrator Decisions

| # | Question | Administrator Decision |
|---|----------|----------------------|
| 1 | Should we evaluate Zipformer/Paraformer as alternative models? | SenseVoice accuracy is acceptable. Keep it. Only evaluate alternatives if chunked SenseVoice is still too slow after benchmarking. |
| 2 | Should we pursue CoreML acceleration on M2? | **Yes**. CoreML on M2, CPU-only fallback on devices without GPU. |
| 3 | Is verbatim (word-by-word) transcription output needed? | No. Only the final transcription file of the whole audio is needed. |
| 4 | Is ffmpeg installed via Homebrew on M2? | **Yes**, confirmed. (Includes AudioToolbox by default.) |

### Key Insight
Chunking SenseVoice into 30s segments should solve the quadratic slowness — each chunk processes independently in ~200ms. This is the highest-impact change with no dependency or model changes required.
