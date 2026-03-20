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
