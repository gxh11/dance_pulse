# server.py
# FastAPI + WebSocket 服务端
#
# 硬约束 (项目约定): 帧数据只在内存中流转, 服务端不落盘任何帧/视频。
#
# 双轨架构 (AGENTS.md):
#   播放轨: yt-dlp 解析 YouTube 直链 -> GET /stream 代理转发 -> 浏览器 <video> 直接播放
#   推理轨: asyncio FFmpeg 子进程从同一 URL 拉流抽帧 -> 内存环形队列 -> 1Hz ONNX 滑窗推理
#
# 接口:
#   GET  /health        -> 服务状态 + 模型信息
#   GET  /resolve       -> yt-dlp 解析 YouTube URL -> 播放直链 + 请求头
#   GET  /stream        -> 直链流式代理 (透传 Range, 支持拖动)
#   POST /feedback      -> 人工反馈标注 (预测正确/有误) -> 追加写 logs/feedback.jsonl
#   WS   /ws/youtube    -> 双轨流式推理 (start/stop 控制, 1Hz pred 推送, 带延迟时间戳)
#   WS   /ws/stream     -> 摄像头推帧模式 (兼容保留): 客户端推二进制帧, 回传 JSON 预测
#
# WebSocket 协议 (/ws/youtube):
#   客户端 -> 服务端 (text/JSON): {"type":"start","url":...} / {"type":"stop"}
#   服务端 -> 客户端 (text/JSON):
#     {"type":"started","play_url":"/stream?u=...","title":...}
#     {"type":"warmup","frames":N}
#     {"type":"pred","pred":"hiphop","confidence":0.97,"probs":{...},
#      "server_ts":<发送时刻>,"frame_ts":<窗口最新帧入队时刻>}   # 端到端延迟打点
#     {"type":"error","msg":...}
import asyncio
import json
import os
import re
import threading
import time
import traceback
import urllib.parse

import numpy as np
import cv2
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from torchvision.models.video import r3d_18
import httpx
import yt_dlp

# ONNX Runtime CUDA 的 DLL (cublas/cudnn) 由 torch wheel 携带, 必须先 import torch
# 再 import onnxruntime, 并把 torch/lib 加入 DLL 搜索路径 (仅 Windows 需要;
# Linux 容器内这些库由镜像构建期 ldconfig 注册, 见 Dockerfile)
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))
from inference_engine import DanceInferenceEngine  # noqa: E402

from stream_pipeline import RingBuffer, normalize_clip
from youtube_pipe import AsyncFramePipe

# 注意: Windows 上 uvicorn --reload 会强制 SelectorEventLoop (不支持 create_subprocess_exec),
# 故 youtube_pipe 用线程泵 + subprocess.Popen, 不依赖事件循环类型 (见 youtube_pipe.py)。

CLASS_NAMES = ["breakdancing", "ballet_jazz", "locking", "hiphop", "popping", "no_dance"]
WEIGHTS = "checkpoints/r3d_dance_6class.pth"
ONNX_PATH = "checkpoints/dance_classifier_6class.onnx"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_FRAMES = 16       # 与训练一致: 16 帧滑窗
WARMUP_FRAMES = 6      # 预热下限: 攒够 6 帧 (~0.75s) 即推理 (左侧 pad 到 16, 与训练 pad 逻辑一致),
                       # 把首条预测的暖机时间从 2s 压到 <1s
INPUT_SIZE = 112       # 与训练一致
RING_CAPACITY = 48     # 每连接环形队列容量 (覆盖旧帧)
PREDICT_EVERY = 4      # [摄像头模式] 每收到几帧推理一次
DEDUP_WINDOW = 0.8     # [YouTube 模式] 帧去重窗口 (秒): 同一画面时刻只推理一次
TARGET_SPACING = 1.0   # [YouTube 模式] 相邻两次推理的视频时间间距 -> 1Hz
STREAM_FPS = 8         # [YouTube 模式] 推理轨抽帧速率
# 极简抓取: 直链解析只需要 player_response 里的渐进式 mp4 URL。跳过观看页 HTML、
# /next 初始数据、ytcfg、JS 播放器、DASH/HLS manifest 五路抓取与全量格式嗅探,
# 单次解析从 ~1.9s (全量嗅探) 压到一次 player API 请求的量级。
YDL_OPTS = {
    # 单流格式: 音视频已合一, 无需 ffmpeg 合并, 也不用落盘
    "format": "best[ext=mp4]/best",
    "quiet": True,
    "no_warnings": True,
    "noplaylist": True,
    "noprogress": True,
    # 单次请求硬上限: 网络抖动时最坏耗时可控, 避免一个僵尸解析拖垮整个线程池
    "socket_timeout": 10,
    "retries": 1,
    # android 客户端直链免签名解密 (少一轮请求);
    # 被 YouTube SABR 实验掐掉直链时由 resolve_video_sync 阶梯回退
    "extractor_args": {"youtube": {
        "player_client": ["android"],
        # manifest 纯属浪费: format=best 只选音视频合一的渐进流, 不会选 DASH 分离流
        "skip": ["dash", "hls"],
        # 砍掉观看页/初始数据/ytcfg/JS 播放器四路额外抓取
        "player_skip": ["webpage", "initial_data", "configs", "js"],
    }},
}
# 中间回退: tv/ios 同样免签名解密, 比 web 快得多。
# 这两个客户端的流偶尔只挂在 HLS manifest 里, 保留 hls 作为无渐进流视频的兜底来源。
YDL_OPTS_TV = {
    **YDL_OPTS,
    "extractor_args": {"youtube": {
        **YDL_OPTS["extractor_args"]["youtube"],
        "player_client": ["tv", "ios"],
        "skip": ["dash"],
    }},
}
YDL_OPTS_WEB = {
    **YDL_OPTS,
    # web 客户端做签名解密 + PO Token 申请, 最慢但格式全, 只做最后兜底。
    # 签名解密依赖 JS 播放器与 ytcfg, 故只砍观看页与 /next 初始数据。
    "extractor_args": {"youtube": {
        **YDL_OPTS["extractor_args"]["youtube"],
        "player_client": ["android", "tv", "web"],
        "skip": ["dash"],
        "player_skip": ["webpage", "initial_data"],
    }},
}
_YDL_TIERS = (YDL_OPTS, YDL_OPTS_TV, YDL_OPTS_WEB)


# 进程内常驻单例: 每个工作线程一套 yt-dlp 实例 (thread-local), 免去每次解析重建
# YoutubeDL 的开销; 且不引入全局锁 —— 不同 URL 的解析按线程并行互不阻塞。
_ydl_local = threading.local()


def _get_ydl(tier: int) -> yt_dlp.YoutubeDL:
    pool = getattr(_ydl_local, "pool", None)
    if pool is None:
        pool = _ydl_local.pool = {}
    ydl = pool.get(tier)
    if ydl is None:
        ydl = pool[tier] = yt_dlp.YoutubeDL(_YDL_TIERS[tier])
    return ydl

app = FastAPI(title="Dance Pulse Streaming API", version="2.0")
MODEL = None          # PyTorch 模型 (摄像头推帧模式 / ONNX 回退)
ENGINE = None         # ONNX 推理引擎 (YouTube 双轨模式, CUDA)
MODEL_READY = threading.Event()   # 模型加载完成标志 (后台线程置位)
# 直链请求头缓存: play_url -> http_headers (googlevideo 需带 Referer/UA, 否则 403)
STREAM_HEADERS: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# 模型: 启动时加载一次, 所有连接共享 (推理无梯度, 线程安全)
# ---------------------------------------------------------------------------
def load_model():
    model = r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    return model.to(DEVICE).eval()


@app.on_event("startup")
def _startup():
    # 模型加载放后台线程: 冷启动 30-60s 期间 /resolve 等轻接口仍可即时响应,
    # 不再让前端在"解析视频中"撞 30s 超时 (请求在 lifespan 启动完成前根本不被处理)
    threading.Thread(target=_load_models, daemon=True).start()
    # 预热 InnerTube visitorData (首个解析请求的 ~5s 冷启动惩罚在此摊掉):
    # 与模型加载并行, 用户第一次贴 URL 时快路径已是稳态 ~0.5s
    threading.Thread(target=_get_visitor_data, daemon=True).start()


def _load_models():
    global MODEL, ENGINE
    MODEL = load_model()
    print(f"[*] PyTorch 模型已加载: {WEIGHTS} @ {DEVICE}")
    try:
        ENGINE = DanceInferenceEngine(ONNX_PATH, threshold=0.45)
        print(f"[*] ONNX 引擎已加载: {ONNX_PATH} @ {ENGINE.session.get_providers()}")
        # 预热: 触发 CUDA context 初始化, 避免首条预测承担 ~0.7s 冷启动
        ENGINE.predict(np.zeros((CLIP_FRAMES, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8))
        print("[*] ONNX 已预热 (CUDA context 就绪)")
    except Exception as e:
        ENGINE = None
        print(f"[!] ONNX 加载失败, 将回退 PyTorch: {e}")
    MODEL_READY.set()


def predict_window(window_u8):
    """window_u8: (16, 112, 112, 3) uint8 -> (pred_name, confidence, probs_dict)"""
    # normalize_clip 返回 numpy (C, T, H, W) float32 -> 转 torch
    x = torch.from_numpy(normalize_clip(window_u8)).unsqueeze(0).to(DEVICE)  # (1, C, T, H, W)
    with torch.no_grad():
        probs = torch.softmax(MODEL(x), dim=1)[0].cpu().numpy()
    pred = int(probs.argmax())
    return CLASS_NAMES[pred], float(probs[pred]), {n: float(p) for n, p in zip(CLASS_NAMES, probs)}


def predict_onnx(window_u8):
    """YouTube 双轨模式的推理入口: 优先 ONNX (CUDA), 失败回退 PyTorch。

    window_u8: (16, 112, 112, 3) uint8 -> (pred_name, confidence, probs_dict)
    """
    if ENGINE is not None:
        try:
            out = ENGINE.predict(window_u8)
            return out["genre"], out["confidence"], out["distribution"]
        except Exception:
            pass
    return predict_window(window_u8)


# ---------------------------------------------------------------------------
# 二进制帧解码 (JPEG/PNG -> resize, 或裸 RGB24 112x112)
# ---------------------------------------------------------------------------
def decode_frame(data: bytes) -> np.ndarray | None:
    if not data:
        return None
    # JPEG (FF D8) / PNG (89 50 4E 47) 走压缩解码
    if data[:2] == b"\xff\xd8" or data[:4] == b"\x89PNG":
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)), cv2.COLOR_BGR2RGB)
    # 裸 RGB24
    if len(data) == INPUT_SIZE * INPUT_SIZE * 3:
        return np.frombuffer(data, np.uint8).reshape(INPUT_SIZE, INPUT_SIZE, 3)
    return None


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL_READY.is_set() else "loading",
        "model": WEIGHTS,
        "device": str(DEVICE),
        "classes": CLASS_NAMES,
        "clip_frames": CLIP_FRAMES,
        "onnx": ENGINE.session.get_providers() if ENGINE is not None else None,
    }


# 解析按 URL 单飞 (single-flight) + 进程内缓存:
# 同 URL 并发请求共享一次 yt-dlp 提取 (排队者直接拿结果); 不同 URL 并行互不
# 阻塞 —— 一个 URL 走 20s 慢路径不会拖垮其它解析。前端超时重试时, 原任务
# 仍在后台跑完并进缓存, 重试秒回缓存, 因此"直接换 URL"慢也能换成功。
_RESOLVE_JOBS: dict[str, dict] = {}   # url -> {"event": Event, "result": dict|None, "error": Exception|None}
_RESOLVE_JOBS_LOCK = threading.Lock()
RESOLVE_CACHE: dict[str, tuple[float, dict]] = {}
RESOLVE_CACHE_TTL = 300.0   # 直链有效期约 6h, 5min 缓存对播放/推理场景足够

# ---------------------------------------------------------------------------
# 极简抓取快路径: 进程内 httpx 常驻单例 (keep-alive 连接池) 直接请求 InnerTube
# player 接口, 一次往返拿直链 —— 绕开 yt-dlp 的全量嗅探/多格式解析开销
# (实测 ~0.7s vs yt-dlp ~2.5-3.2s)。ANDROID 客户端直链免签名解密, 取单流 MP4。
# 关键: 必须带 visitorData + 完整 client/user/request context, 否则 YouTube 返回
# 400 "Precondition check failed" (实测踩坑)。
# 失败 (SABR 掐直链/区域限制/接口变更) 自动回退 yt-dlp 阶梯, 行为不变。
# ---------------------------------------------------------------------------
_INNERTUBE_CLIENT = httpx.Client(
    timeout=httpx.Timeout(6.0),
    limits=httpx.Limits(max_keepalive_connections=4),
)
_INNERTUBE_URL = "https://www.youtube.com/youtubei/v1/player?prettyPrint=false"
_INNERTUBE_KEY = "AIzaSyA8eiZmM1FaDVjRy-df2KTyQ_vz_yYM39w"   # Android 客户端公开 InnerTube key
_INNERTUBE_UA = "com.google.android.youtube/20.10.38 (Linux; U; Android 11) gzip"
_INNERTUBE_CLIENT_VERSION = "20.10.38"
_YT_ID_RE = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/)([\w-]{11})")
_VISITOR_RE = re.compile(r'"VISITOR_DATA":"([\w%=-]+)"')
# 优先级: 单流 MP4 (音视频合一, 无需 ffmpeg 合流) 高清优先
_INNERTUBE_ITAG_PRIORITY = (22, 18, 59, 13)   # 720p/360p/480p/240p mp4

# visitorData 是会话令牌 (有效期数小时), 拉一次全局复用, 摊销后稳态解析 ~0.5s
_VISITOR_CACHE: dict = {"data": None, "ts": 0.0}
_VISITOR_TTL = 1800.0
_VISITOR_LOCK = threading.Lock()


def _get_visitor_data() -> str | None:
    with _VISITOR_LOCK:
        if _VISITOR_CACHE["data"] and time.time() - _VISITOR_CACHE["ts"] < _VISITOR_TTL:
            return _VISITOR_CACHE["data"]
        try:
            r = _INNERTUBE_CLIENT.get(
                "https://www.youtube.com/",
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            m = _VISITOR_RE.search(r.text)
            _VISITOR_CACHE["data"] = m.group(1) if m else None
            _VISITOR_CACHE["ts"] = time.time()
            return _VISITOR_CACHE["data"]
        except Exception:
            return _VISITOR_CACHE["data"]


def _resolve_via_innertube(url: str) -> dict | None:
    """单次 InnerTube 请求拿直链; 拿不到返回 None (走 yt-dlp 阶梯兜底)。"""
    m = _YT_ID_RE.search(url)
    if not m:
        return None
    visitor = _get_visitor_data()
    try:
        r = _INNERTUBE_CLIENT.post(
            _INNERTUBE_URL + f"&key={_INNERTUBE_KEY}",
            headers={
                "User-Agent": _INNERTUBE_UA,
                "Content-Type": "application/json",
                "X-Youtube-Client-Name": "3",
                "X-Youtube-Client-Version": _INNERTUBE_CLIENT_VERSION,
                **({"X-Goog-Visitor-Id": visitor} if visitor else {}),
            },
            json={
                "videoId": m.group(1),
                "contentCheckOk": True,
                "racyCheckOk": True,
                "context": {
                    "client": {
                        "clientName": "ANDROID",
                        "clientVersion": _INNERTUBE_CLIENT_VERSION,
                        "androidSdkVersion": 30,
                        "hl": "en", "gl": "US",
                        "utcOffsetMinutes": 0,
                        "clientScreen": "EMBED",
                        **({"visitorData": visitor} if visitor else {}),
                    },
                    "user": {"lockedSafetyMode": False},
                    "request": {"useSsl": True},
                },
            },
        )
        data = r.json()
    except Exception:
        return None

    if (data.get("playabilityStatus", {}).get("status") or "") != "OK":
        return None
    formats = (data.get("streamingData") or {}).get("formats") or []
    fmt = None
    for itag in _INNERTUBE_ITAG_PRIORITY:
        fmt = next((f for f in formats if f.get("itag") == itag and f.get("url")), None)
        if fmt:
            break
    if fmt is None:
        return None   # 该客户端无直链 (SABR) -> 回退

    return {
        "title": (data.get("videoDetails") or {}).get("title"),
        "url": fmt["url"],
        "ext": (fmt.get("mimeType") or "video/mp4").split(";")[0].split("/")[-1],
        # android 直链需带原生 UA/Referer, 否则 googlevideo 403
        "http_headers": {"User-Agent": _INNERTUBE_UA,
                         "Referer": "https://www.youtube.com/"},
    }


def resolve_video_sync(url: str) -> dict:
    """解析 YouTube URL -> 播放直链 (不下载, 零落盘)。

    极简抓取快路径 InnerTube (~0.7s); 失败回退 yt-dlp 阶梯:
    android (免签名解密) -> tv/ios (免签名) -> web (签名+PO Token, 最慢兜底)。
    """
    cached = RESOLVE_CACHE.get(url)
    if cached and time.time() - cached[0] < RESOLVE_CACHE_TTL:
        return cached[1]

    with _RESOLVE_JOBS_LOCK:
        job = _RESOLVE_JOBS.get(url)
        owner = job is None
        if owner:
            job = {"event": threading.Event(), "result": None, "error": None}
            _RESOLVE_JOBS[url] = job

    if not owner:
        # 同 URL 的解析在途: 等它完成, 结果进缓存直接复用
        if not job["event"].wait(timeout=60):
            raise RuntimeError("解析仍在进行, 请稍候重试")
        if job["error"] is not None:
            raise RuntimeError(f"解析失败: {job['error']}")
        return job["result"]

    try:
        # 快路径优先: InnerTube 一次往返拿直链 (~0.7s)
        result = _resolve_via_innertube(url)
        if result is None:
            # 兜底: yt-dlp 客户端阶梯 android -> tv/ios -> web
            info, last_err = None, None
            for tier in range(len(_YDL_TIERS)):
                try:
                    info = _get_ydl(tier).extract_info(url, download=False)
                    if info.get("url"):
                        break
                    last_err = "该客户端无直链 (SABR 限制)"
                except Exception as e:
                    info, last_err = None, str(e)
            if not info or not info.get("url"):
                raise RuntimeError(f"解析失败: {last_err}")

            result = {
                "title": info.get("title"),
                "url": info["url"],
                "ext": info.get("ext", ""),
                "http_headers": info.get("http_headers", {}) or {},
            }
        if len(RESOLVE_CACHE) > 16:
            RESOLVE_CACHE.clear()
        RESOLVE_CACHE[url] = (time.time(), result)
        job["result"] = result
        return result
    except Exception as e:
        job["error"] = e
        raise
    finally:
        with _RESOLVE_JOBS_LOCK:
            _RESOLVE_JOBS.pop(url, None)
        job["event"].set()


@app.get("/resolve")
async def resolve(url: str = Query(..., description="YouTube 视频 URL")):
    """解析 YouTube URL -> 播放直链 (HLS/MP4)。请求头缓存供 /stream 代理与推理轨使用。"""
    try:
        info = await asyncio.to_thread(resolve_video_sync, url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"解析失败: {e}")
    if len(STREAM_HEADERS) > 10:
        STREAM_HEADERS.clear()
    STREAM_HEADERS[info["url"]] = info["http_headers"]
    return {"title": info["title"], "play_url": info["url"], "ext": info["ext"]}


# 复用 HTTP 连接池: 每次 Range 请求(拖动/首块)都省去 TLS 握手, 直接关系到 TTFF <500ms
_HTTP = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=120.0),
                          follow_redirects=True, limits=httpx.Limits(max_keepalive_connections=8))


@app.on_event("shutdown")
async def _shutdown_http():
    await _HTTP.aclose()


# googlevideo 403 兜底头: STREAM_HEADERS 未命中 (服务重启/缓存淘汰) 时使用,
# 保证 sessionStorage 里的旧 play_url 走 /stream 仍可播放
DEFAULT_STREAM_HEADERS = {"User-Agent": _INNERTUBE_UA, "Referer": "https://www.youtube.com/"}


@app.get("/stream")
async def stream_proxy(request: Request, u: str = Query(..., description="urlencode 后的播放直链")):
    """直链流式代理: 透传 Range (支持拖动), 带上 Referer/UA 绕过 googlevideo 403。

    206 流式直通三要点 (原生 <video> 快速起播的关键):
    1. 只透传浏览器的 Range —— 浏览器的 Referer (localhost) 不能转给 googlevideo,
       UA/Referer 统一用解析时存下的客户端头; 缓存未命中回退默认 YouTube 头,
       服务重启后 sessionStorage 里的旧 play_url 也能播。
    2. content-range / content-length / accept-ranges 原样透传, 浏览器据此识别 206 分段。
    3. aiter_raw 字节级直通: 上游来多少吐多少。aiter_bytes 会攒满 chunk_size 才吐,
       首字节被累积缓冲拖慢, 与 <100ms 起播相悖。
    """
    play_url = urllib.parse.unquote(u)
    fwd = {k: v for k, v in STREAM_HEADERS.get(play_url, DEFAULT_STREAM_HEADERS).items()
           if k.lower() in ("user-agent", "referer")}
    if "range" in request.headers:
        fwd["range"] = request.headers["range"]

    try:
        upstream = await _HTTP.send(
            _HTTP.build_request("GET", play_url, headers=fwd), stream=True)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"上游拉流失败: {e}")

    pass_headers = {k: v for k, v in upstream.headers.items()
                    if k.lower() in ("content-type", "content-length", "content-range",
                                     "accept-ranges", "content-encoding")}

    async def gen():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(gen(), status_code=upstream.status_code, headers=pass_headers)


# ---------------------------------------------------------------------------
# 人工反馈标注 (human-in-the-loop): 预测正确/有误 -> 追加写 JSONL 日志
# ---------------------------------------------------------------------------
FEEDBACK_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "feedback.jsonl")


class FeedbackIn(BaseModel):
    label: str                          # "correct" | "wrong"
    pred: str | None = None             # 模型当时的预测
    confidence: float | None = None
    probs: dict | None = None
    frame_ts: float | None = None       # 该预测对应的画面事件时刻
    video_url: str | None = None
    title: str | None = None
    client_ts: float | None = None      # 前端打点时刻


@app.post("/feedback")
def feedback(fb: FeedbackIn):
    if fb.label not in ("correct", "wrong"):
        raise HTTPException(status_code=422, detail="label 必须是 correct 或 wrong")
    entry = {"ts": time.time(), **fb.model_dump()}
    os.makedirs(os.path.dirname(FEEDBACK_LOG), exist_ok=True)
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# YouTube 双轨模式: 推理轨 (asyncio FFmpeg 拉流 -> 环形队列 -> 1Hz ONNX 推理)
# ---------------------------------------------------------------------------
async def _producer_task(pipe: AsyncFramePipe, ring: RingBuffer):
    """推理轨: 逐帧拉流入环形队列 (旧帧覆盖), 全程内存。"""
    async for frame, ts in pipe.frames():
        ring.append(frame, ts)


async def _run_producer_task(pipe: AsyncFramePipe, ring: RingBuffer, ws: WebSocket):
    """包装: 拉流任务异常必须回报客户端, 不能静默死亡。"""
    try:
        await _producer_task(pipe, ring)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        try:
            await ws.send_text(json.dumps({"type": "error", "msg": f"拉流任务异常: {e!r}"}))
        except Exception:
            pass
        print(f"[!] producer 任务异常: {e!r}\n{tb}")


async def _run_inference_task(ws: WebSocket, ring: RingBuffer, producer: asyncio.Task):
    """包装: 推理任务异常必须回报客户端, 不能静默死亡。"""
    try:
        await _inference_task(ws, ring, producer)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        try:
            await ws.send_text(json.dumps({"type": "error", "msg": f"推理任务异常: {e!r}"}))
        except Exception:
            pass
        print(f"[!] inference 任务异常: {e!r}\n{tb}")


async def _inference_task(ws: WebSocket, ring: RingBuffer, producer: asyncio.Task):
    """推理循环: 1Hz 滑窗推理, pred 消息带 server_ts / frame_ts 供前端算端到端延迟。

    节流口径 (关键): 用"视频时间" (窗口最新帧的入队间隔) 去重并按 TARGET_SPACING
    控制节奏, 而非固定 sleep(1s)。否则 ffmpeg 一秒钟会把同一批帧的最新画面重复
    推理多次 —— frame_ts 停滞, E2E = now - frame_ts 虚高, 既不真实也浪费推理。
    去重窗口取 0.8s: ffmpeg 每 ~1.25s 才产出新的 8fps 框, 低于此值即为同一画面。
    """
    last_frames = -1
    last_frame_ts = 0.0   # 上次推理所用的视频时刻
    last_pred_ts = 0.0    # 上次推理的墙钟时刻
    while True:
        n = len(ring)
        if n < WARMUP_FRAMES:
            if n != last_frames:  # 帧数变化才播报, 避免刷屏
                last_frames = n
                await ws.send_text(json.dumps({"type": "warmup", "frames": n}))
            if producer.done():
                break  # 流在预热前就结束
            await asyncio.sleep(0.1)  # 预热期快速探测, 满窗立即推理
            continue

        frame_ts = ring.get_window_ts(CLIP_FRAMES)[1]
        now = time.time()
        video_advanced = (frame_ts - last_frame_ts) >= DEDUP_WINDOW
        # 间隔 = 墙钟与视频时间的较大者, 两者任一没到 1Hz 都不推理
        spacing_met = (now - last_pred_ts) >= TARGET_SPACING
        if video_advanced and spacing_met:
            window, frame_ts = ring.get_window_ts(CLIP_FRAMES)  # frame_ts = 画面事件时刻
            pred, conf, probs = await asyncio.to_thread(predict_onnx, window)
            await ws.send_text(json.dumps({
                "type": "pred",
                "pred": pred,
                "confidence": conf,
                "probs": probs,
                "server_ts": time.time(),
                "frame_ts": frame_ts,
            }))
            last_frame_ts = frame_ts
            last_pred_ts = time.time()
            if producer.done():
                break  # 流已结束, 帧时间戳不再推进, 停止推理避免 E2E 失真
        await asyncio.sleep(0.1)


async def _start_session(ws: WebSocket, url: str,
                         play_url: str | None = None, title: str | None = None,
                         start_at: float = 0.0) -> dict | None:
    """起推理轨 producer + 1Hz inference 两个任务。失败返回 None。

    客户端可携带已解析的 play_url (前端已通过 /resolve 预解析), 跳过 1.9s 的
    yt-dlp 解析, 使其不占用 TTFP / TTFF 关键路径; 请求头取自 /resolve 的缓存。
    start_at: 从视频第几秒开始推理 (前端进度条拖动时同步跳转)。
    """
    if not MODEL_READY.is_set():
        await ws.send_text(json.dumps({"type": "error", "msg": "模型仍在加载中, 请稍候重试"}))
        return None
    headers = STREAM_HEADERS.get(play_url) if play_url else None
    if not (play_url and headers):
        try:
            info = await asyncio.to_thread(resolve_video_sync, url)
        except Exception as e:
            await ws.send_text(json.dumps({"type": "error", "msg": f"解析失败: {e}"}))
            return None
        play_url = info["url"]
        headers = info["http_headers"]
        title = info.get("title")
        if len(STREAM_HEADERS) > 10:
            STREAM_HEADERS.clear()
        STREAM_HEADERS[play_url] = headers
    proxy_url = "/stream?u=" + urllib.parse.quote(play_url, safe="")
    await ws.send_text(json.dumps({
        "type": "started", "play_url": proxy_url, "title": title,
        "ext": "m3u8" if "m3u8" in play_url else "mp4"}))

    ring = RingBuffer(capacity=RING_CAPACITY)
    pipe = AsyncFramePipe(play_url, headers, fps=STREAM_FPS, start_at=start_at)
    producer = asyncio.create_task(_run_producer_task(pipe, ring, ws))
    infer = asyncio.create_task(_run_inference_task(ws, ring, producer))
    return {"pipe": pipe, "tasks": [producer, infer]}


async def _stop_session(session: dict | None):
    if not session:
        return
    for t in session["tasks"]:
        t.cancel()
    for t in session["tasks"]:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
    await session["pipe"].close()


@app.websocket("/ws/youtube")
async def ws_youtube(ws: WebSocket):
    """YouTube 双轨流式推理。控制: start / stop; 推送: started / warmup / pred / error。"""
    await ws.accept()
    print("[*] YouTube 模式客户端已连接")
    session = None
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            text = message.get("text")
            if not text:
                continue
            try:
                cmd = json.loads(text)
            except json.JSONDecodeError:
                continue

            await _stop_session(session)
            session = None

            if cmd.get("type") == "start":
                session = await _start_session(
                    ws, cmd.get("url", ""),
                    play_url=cmd.get("play_url"), title=cmd.get("title"),
                    start_at=float(cmd.get("start_at") or 0))
            elif cmd.get("type") == "stop":
                await ws.send_text(json.dumps({"type": "stopped"}))
    except WebSocketDisconnect:
        pass
    finally:
        await _stop_session(session)
        print("[*] YouTube 模式客户端断开")


# ---------------------------------------------------------------------------
# WebSocket 流式推理 (摄像头推帧模式, 兼容保留)
# ---------------------------------------------------------------------------
@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    if not MODEL_READY.is_set():
        await ws.send_text(json.dumps({"type": "error", "msg": "模型仍在加载中, 请稍候重试"}))
        await ws.close()
        return
    ring = RingBuffer(capacity=RING_CAPACITY)
    frame_id = 0
    print("[*] WebSocket 客户端已连接")
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break

            data = message.get("bytes")
            if data is None:
                # 允许客户端发 text 控制指令
                text = message.get("text")
                if text == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
                elif text == "reset":
                    ring.clear()
                    frame_id = 0
                    await ws.send_text(json.dumps({"type": "reset_ok"}))
                continue

            frame = decode_frame(data)
            if frame is None:
                await ws.send_text(json.dumps(
                    {"type": "error", "msg": "无法解码帧 (需 JPEG/PNG 或 112x112x3 裸 RGB)"}))
                continue

            ring.append(frame)          # O(1) 入队, 旧帧被覆盖, 全程内存
            frame_id += 1

            # 队列攒够一窗后, 按 PREDICT_EVERY 节流推理
            if len(ring) >= CLIP_FRAMES and frame_id % PREDICT_EVERY == 0:
                window = ring.get_window(CLIP_FRAMES)   # (16, 112, 112, 3)
                # 推理在默认线程池执行, 不阻塞事件循环收帧
                pred, conf, probs = await asyncio.to_thread(predict_window, window)
                await ws.send_text(json.dumps({
                    "type": "pred",
                    "frame_id": frame_id,
                    "pred": pred,
                    "confidence": conf,
                    "probs": probs,
                    "warmup": False,
                }))
            elif len(ring) < CLIP_FRAMES:
                await ws.send_text(json.dumps({
                    "type": "pred", "frame_id": frame_id, "warmup": True,
                    "msg": f"预热中 {len(ring)}/{CLIP_FRAMES}",
                }))
    except WebSocketDisconnect:
        pass
    finally:
        print(f"[*] WebSocket 客户端断开 (共收 {frame_id} 帧)")


# 前端静态站点 (vite 构建产物 web/dist; 容器内由 Dockerfile 生成):
# 单进程同时服务 API 与页面, 开发时仍走 5173 的 vite dev server。
# 必须最后挂载: mount("/") 兜底所有未匹配路径, 需排在全部 API/WS 路由之后。
WEB_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "dist")
if os.path.isdir(WEB_DIST):
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8001)
