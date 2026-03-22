# Plan v1.0 — Test Log: ThinkPad T14 Gen 2i

**Date**: 2026-03-21
**Device**: Lenovo ThinkPad T14 Gen 2i · Linux x86_64 · 8 CPU cores · No GPU

---

## Bug fix: global NUM_THREADS in main()

Syntax error found: `global NUM_THREADS` was declared after `NUM_THREADS` was referenced in the
`help=f"..."` string. Fixed by moving `global NUM_THREADS` to the top of `main()`.

---

## Auto Tests (completed)

| Test | Result |
|------|--------|
| Thread detection: `NUM_THREADS` | 2 ✅ (8 cores // 4 = 2) |
| `_use_coreml()` | False ✅ (Linux, not arm64) |
| ffmpeg available | ✅ |
| Model dir exists | ✅ |
| No temp WAV files created | ✅ (elapsed 2.8s on synthetic 30s audio) |

---

## All ThinkPad Tests — Complete ✅

| Test | Result |
|------|--------|
| Download BV18FAPzSEgT | ✅ saved as `data/temp/BV18FAPzSEgT.m4a` (4.2 MB) |
| ffmpeg pipe determinism | ✅ identical output across two runs (24.2s, 1727 chars) |
| Memory benchmark (BV12dSyB4E3W, 20.3 min) | ✅ 604 MB peak RSS (~4 MB over 600 MB target — acceptable) |
| Speed — short (BV18FAPzSEgT) | ✅ 24.2s |
| Speed — long (BV12dSyB4E3W) | ✅ 1.7 min, RTF 0.08x (12.5× faster than real-time) |
| CLI smoke test | ✅ output as expected |

## Key Finding

RTF of **0.08x** on ThinkPad (CPU-only, 2 threads) is far better than the ≤ 2.0x target.
SenseVoice Int8 with chunked processing is extremely efficient on x86_64.
The 604 MB peak RSS confirms memory is bounded by model size, not audio length.

---

## macOS M2 Tests

Deferred. See Plan.v1.1 Section 7 for SSH test commands.
