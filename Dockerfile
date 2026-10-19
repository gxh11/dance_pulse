# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Dance Pulse — 单镜像一键部署
#   CPU: docker compose up -d --build
#   GPU: docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
#
# 镜像内容: FastAPI 服务端 (8001) + 前端静态站点 (vite 构建产物)
#           + FFmpeg (imageio-ffmpeg 自带二进制, 推理轨抽帧零落盘)
# 权重/ONNX 由 docker-compose 从 ./checkpoints 挂载, 不打包进镜像
# ---------------------------------------------------------------------------

# ---------- Stage 1: 前端构建 (vite) ----------
FROM node:20-slim AS web
WORKDIR /build
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# ---------- Stage 2: 推理服务 ----------
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Python 依赖 (torch/torchvision 由 cu128 源覆盖, 与本地训练环境一致)
COPY requirements.txt .
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128

# onnxruntime-gpu 运行期需 libcublas/libcudnn 等 CUDA 动态库, 但它们只随
# torch/nvidia pip 包存在于 site-packages 内 (非系统路径); 把这些目录注册进
# ldconfig, 否则 ORT 加载 CUDAExecutionProvider 时找不到 .so 静默回退 CPU
RUN python -c "import glob, os, site, torch; \
    libs = [os.path.join(os.path.dirname(torch.__file__), 'lib')]; \
    libs += sorted(glob.glob(os.path.join(site.getsitepackages()[0], 'nvidia', '*', 'lib'))); \
    open('/etc/ld.so.conf.d/or-cuda.conf', 'w').write('\n'.join(libs) + '\n')" && ldconfig

# 服务端代码 (仅部署所需 4 个模块, 训练/数据脚本不进镜像)
COPY server.py inference_engine.py stream_pipeline.py youtube_pipe.py ./

# 前端构建产物 -> /app/web/dist (server.py 检测到后以 StaticFiles 挂载于 "/")
COPY --from=web /build/dist ./web/dist

# opencv-python 的 Linux 动态链接依赖 (libxcb/libGL 等), 缺失时 import cv2 报 libxcb.so.1;
# ffmpeg 用发行版构建 (imageio-ffmpeg 的静态二进制在容器内拉 HTTPS 流会段错误)
# (放在末尾以免打断 pip 层缓存)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8001

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
