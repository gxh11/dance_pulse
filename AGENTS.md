python=11.3
dance-pulse              C:\Users\85487\anaconda3\envs\dance-pulse


# Project Overview: Real-Time Dance Genre Classification System (Dance-Pulse)

## 1. Mission & Objective
Building an end-to-end, low-latency, full-stack AI web application for a technical interview take-home assessment. The application allows users to paste a YouTube URL, play the video with sub-500ms initial latency, and continuously infer the active dance genre at 1 Hz using a spatiotemporal deep learning model.

## 2. Core Architecture & Pipeline
- **Dual-Track Streaming (Zero Disk I/O)**:
  - **Playback Track**: Resolves direct streaming URLs (HLS/MP4) via `yt-dlp` for direct client playback in the browser.
  - **Inference Track**: Streams video frames via an `asyncio` FFmpeg sub-process pipe (`pipe:1`, `rawvideo`, `rgb24` at 8 FPS, 112x112) directly into an in-memory ring buffer (`collections.deque(maxlen=16)`). No intermediate frame files or video files are written to disk.
- **Model & Inference Engine**:
  - Spatiotemporal 3D-CNN (`r3d_18`) with 2-second temporal context window (16 frames, shape: `(1, 3, 16, 112, 112)`).
  - Fine-tuned on multi-class dance datasets covering 5 target genres plus a threshold-based fallback mechanism (`Unknown / No Dance`).
  - Served locally via PyTorch CUDA execution (migrating to ONNX Runtime for deployment).
- **Communication & Transport**:
  - Backend: FastAPI service running on port `8001` providing REST endpoints and a bidirectional WebSocket (`ws://127.0.0.1:8001/ws`) to push inference payloads (genre labels, confidence scores, class distributions) to the client.
  - Frontend: Vite-powered SPA (React/Vue) running on port `5173` with real-time video sync, metrics overlay, and feedback controls.

## 3. Current Implementation Status
- [x] In-memory zero-disk streaming pipeline verified (`stream_pipeline.py`).
- [x] Spatiotemporal model checkpoint trained and loaded: `checkpoints/r3d_dance_best.pth` on CUDA.
- [x] FastAPI server up and running on port `8001` (`server.py`).
- [x] Frontend-backend WebSocket link and UI overlay integration (dual-track: playback via `/stream` + inference via `WS /ws/youtube`; latency verified: TTFF 52ms / E2E 134ms).
- [x] Human-in-the-loop prediction flagging and feedback logging endpoint (`POST /feedback` -> `logs/feedback.jsonl`; frontend accurate/wrong buttons on each prediction).
- [x] ONNX Runtime export and latency benchmark comparison (`checkpoints/dance_classifier.onnx`, CUDA ~10ms/step).
- [x] Docker and Docker Compose containerization for single-command deployment (`docker compose up -d --build` → :8001 serving API + frontend; GPU via `docker-compose.gpu.yml` override).

```

---

### Immediate Execution Checklist

1. **Verify Frontend Endpoint Mapping**:
In your frontend repository (`frontend/src`), confirm the WebSocket target is pointing to:
```javascript
const WS_URL = "ws://127.0.0.1:8001/ws";

```


2. **Launch Dev Client**:
Open a separate terminal window and execute:
```powershell
cd frontend
npm run dev

```


3. **Validate End-to-End Handshake**:
Open `http://localhost:5173` in your browser, enter a test video URL, and verify the backend terminal on port 8001 displays an active WebSocket connection with streaming frame logs.