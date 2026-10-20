python=3.11.3
dance-pulse              C:\Users\85487\anaconda3\envs\dance-pulse


# Project Overview: Real-Time Dance Genre Classification System (Dance-Pulse)

## 1. Mission & Objective
Building an end-to-end, low-latency, full-stack AI web application for a technical interview take-home assessment. The application allows users to paste a YouTube URL, play the video with sub-500ms initial latency (TTFF), and continuously infer the active dance genre at 1 Hz using a spatiotemporal deep learning model. Latency targets: TTFF < 500ms / E2E < 1000ms / TTFP <= 4s (measured: 37-58ms / 84-160ms / 1.8-3s).

## 2. Core Architecture & Pipeline
- **Dual-Track Streaming (Zero Disk I/O)**:
  - **Playback Track**: URL is pre-resolved as soon as the user types / pastes / blurs the input field (no click needed). Resolution: InnerTube fast path (single player request, signature-free clients, single-stream MP4) with `yt-dlp` ladder fallback (android -> tv/ios -> web, thread-local persistent `YoutubeDL` singletons, minimal opts skipping manifest/webpage/JS player sniffing). 5-min TTL cache + single-flight dedupe; visitorData pre-warmed at startup. Playback via `GET /stream` proxy: HTTP 206 Range pass-through (byte-level `aiter_raw`, upstream client headers only — browser Referer must NOT be forwarded).
  - **Inference Track**: Streams video frames via an FFmpeg sub-process (`pipe:1`, `rawvideo`, `rgb24` at 8 FPS, 112x112) into an in-memory ring buffer (48-capacity), 16-frame sliding window (2s context). Uses `subprocess.Popen` + reader thread (works on any event loop / `uvicorn --reload`). No intermediate frame or video files are written to disk.
- **Model & Inference Engine**:
  - Spatiotemporal 3D-CNN (`r3d_18`), input shape `(1, 3, 16, 112, 112)`.
  - 6-class: 5 dance genres (breakdancing / ballet_jazz / locking / hiphop / popping) + `no_dance`; low-confidence (< 0.45) falls back to `Unknown / No Dance`.
  - ONNX Runtime CUDA primary (`checkpoints/dance_classifier_6class.onnx`, ~10ms/step) with automatic PyTorch fallback; checkpoint `checkpoints/r3d_dance_6class.pth`.
- **Communication & Transport**:
  - Backend: FastAPI on port `8001`: `GET /health`, `GET /resolve`, `GET /stream`, `POST /feedback`, `WS /ws/youtube` (start/stop, 1Hz predictions with latency timestamps), `WS /ws/stream` (legacy camera-push mode).
  - Frontend: Vite SPA (vanilla JS + hls.js) in `web/`; dev server on `5173` (proxies to 8001), production served from `web/dist` by the same server. WS target is dynamic: `ws(s)://<location.host>/ws/youtube`. TTFF measured at the frame-presentation timestamp via `requestVideoFrameCallback`'s `now` (double-rAF / `loadeddata` fallbacks for non-rVFC browsers).

## 3. Current Implementation Status
- [x] In-memory zero-disk streaming pipeline verified (`stream_pipeline.py`).
- [x] 6-class model trained and loaded: `checkpoints/r3d_dance_6class.pth` (test acc 86.3%, grouped anti-leak splits; no_dance F1 = 1.00).
- [x] FastAPI server up and running on port `8001` (`server.py`).
- [x] Frontend-backend WebSocket link and UI overlay (dual-track: playback via `/stream` + inference via `WS /ws/youtube`).
- [x] Human-in-the-loop feedback logging (`POST /feedback` -> `logs/feedback.jsonl`; accurate/wrong buttons per prediction).
- [x] ONNX Runtime export and benchmark (`checkpoints/dance_classifier_6class.onnx`, CUDA ~10ms/step).
- [x] Docker / Docker Compose single-command deployment (`docker compose up -d --build` -> :8001 serves API + frontend; GPU via `docker-compose.gpu.yml` override).
- [x] Cold-start latency surgery: InnerTube resolve fast path + cache (~0ms hit, 358-642ms fresh vs ~2s full yt-dlp sniff), `/stream` 206 byte-level pass-through fix, frontend prewarm (debounced input / immediate paste / blur) + media prefetch, precise TTFF timing (rVFC `now`). Verified: TTFF 37-58ms / E2E 84-160ms / TTFP 1.8-3s.

## 4. Key Constraints
- Never write full video files or frame images to disk — pipe through stdin/stdout and in-memory buffers only.
- Dataset splits must be grouped by source video / choreography ID (all clips of one source belong to exactly one of train/val/test); `make_splits.py` asserts zero cross-set group overlap.
- Do not add external YouTube clips into `dance_dataset_categorized` without explicit user approval.
- On Windows use `python -m uvicorn server:app --port 8001` (run as `server.py` also works) — never `--reload` with asyncio subprocess; `youtube_pipe.py` already handles event-loop compatibility.

---

### Immediate Execution Checklist

1. **Verify Frontend WS Mapping** (`web/main.js`):
```javascript
const WS_URL = wsProto + location.host + '/ws/youtube'   // dynamic host; ws://127.0.0.1:8001 when served from :8001
```

2. **Launch Dev Stack**:
```powershell
# backend (port 8001)
conda activate dance-pulse
python server.py            # or: python -m uvicorn server:app --port 8001

# frontend (port 5173, vite dev server proxying to 8001)
cd web
npm run dev
```
Note: `npm run build` regenerates `web/dist` (what the :8001 single-command deployment serves) — rebuild after editing frontend sources.

3. **Validate End-to-End Handshake**:
Open `http://localhost:8001` (or `http://localhost:5173` in dev), paste a YouTube URL (pre-resolve should fire without clicking), click **"播放并推理"**, and verify: video plays, TTFF/E2E/TTFP metrics render green, and the backend shows an active `/ws/youtube` connection with 1Hz frame logs.
