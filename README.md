# Dance Pulse — 实时舞种识别 (Real-Time Dance Genre Classification)

粘贴一个 YouTube 链接，浏览器即时播放（**首帧 < 500 ms**），后端持续以 **1 Hz** 推理当前舞种并通过 WebSocket 推送（**端到端 < 1 s**）。

支持 5 个舞种 + 非舞蹈识别（6 分类模型）：

`breakdancing` · `ballet_jazz` · `locking` · `hiphop` · `popping` · **`no_dance`（非舞蹈）**

非舞蹈内容（演讲/烹饪/课堂/街拍等 15 类真实场景负样本训练）直接判为"非舞蹈"；另有低置信度阈值回退兜底。

---

## 快速开始

### 方式一：Docker 一键部署（推荐）

```bash
docker compose up -d --build        # -> http://localhost:8001 (API + 前端同端口)

# GPU 版 (需 NVIDIA 驱动 + NVIDIA Container Toolkit)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

docker compose down
```

- **单镜像**：FastAPI 服务端 + vite 构建的前端静态站，单端口同时服务 API / WebSocket / 页面
- **权重不入镜像**：`./checkpoints` 以只读 volume 挂载（`r3d_dance_6class.pth` + `dance_classifier_6class.onnx`）
- 基础镜像 `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`；构建期把 torch/nvidia 的 CUDA 库注册进 ldconfig，保证 `onnxruntime-gpu` 加载 CUDAExecutionProvider（无 GPU 自动回退 CPU）
- 健康检查探 `GET /health`（首次启动含模型加载 + 预热，`start_period` 120s）
- 运行时唯一 I/O 为推理轨 FFmpeg 内存管道，**零落盘约束在容器内不变**

### 方式二：本地开发

```powershell
# 后端 (端口 8001)
conda activate dance-pulse
python -m uvicorn server:app --port 8001
# 注意: 勿用 --reload —— Windows 下文件监控重载会弄丢 worker 子进程,
# 父进程占着端口导致所有请求静默挂起 (前端表现为"解析视频中"永远转圈)

# 前端 (端口 5173, vite 开发服务器, 代理到 8001)
cd web
npm install
npm run dev
```

打开 http://localhost:5173，粘贴 YouTube URL，点击 **"播放并推理"**。

**Python 环境**（[requirements.txt](./requirements.txt)）：

```bash
# torch/torchvision 需 CUDA 12.8 构建 (RTX 5060 / Blackwell), 必须用官方 cu128 源
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

---

## 系统架构：双轨流式（全程零落盘）

```
                        ┌──────────────── 播放轨 (低延迟) ────────────────┐
YouTube URL ──▶ yt-dlp 解析直链 ──▶ GET /stream 代理 (透传 Range) ──▶ <video> 直接播放
      │
      └──────────────── 推理轨 (1 Hz) ──────────────────────────────┐
          FFmpeg 子进程拉流 (rawvideo/rgb24, 8 FPS, 112x112)         │
                    │ pipe:1 (内存管道, 零临时文件)                   ▼
                    └──▶ RingBuffer 环形队列 (容量 48, 覆盖旧帧)
                                    │ 每 16 帧滑窗
                                    ▼
                    ONNX Runtime (CUDA, ~10 ms/次) ──▶ WS 推送 pred + 延迟时间戳
```

- **播放轨**：`/resolve` 解析直链 → `/stream` 流式代理（带 Referer/UA 绕过 googlevideo 403）→ 浏览器 `<video>` 播放，支持拖动（Range 透传）
- **推理轨**：FFmpeg 从同一 URL 拉流抽帧 → 内存 `RingBuffer` → 16 帧滑窗（约 2 秒上下文）→ 3D-CNN 推理
- **1 Hz 节流**：视频时间去重（0.8s）+ 墙钟间隔（1.0s）双条件，真实 1 次/秒而非重复推理同帧
- **零落盘**：抽帧走 FFmpeg stdout 管道直接进内存，全程不写任何视频/帧文件

---

## 延迟指标（实测达标）

| 指标 | 口径 | 目标 | 实测 |
|---|---|---|---|
| **TTFF** 首帧 | 点击播放 → 首个渲染帧 | < 500 ms | **52 ms** |
| **E2E** 端到端 | 服务端抽帧入队 `frame_ts` → 前端渲染完预测 | < 1000 ms | **134 ms** |
| **TTFP** 首条预测 | 点击 → 第一条预测（含解析 + 起流 + 暖机） | ≤ 4 s | **1.8 s** |

关键优化：

- **TTFF**：URL 一输入即预解析（`/resolve`）并**预缓冲媒体**，点击后只剩 `play()` → 出帧；首帧用 `requestVideoFrameCallback` 精确打点
- **TTFP**：WS `start` 携带已解析直链（跳过重复 yt-dlp）；启动期 ONNX 预热（冷启动 453 ms → 热 17 ms）；预热窗 6 帧即出首条预测（左侧 pad 到 16，与训练 pad 逻辑一致）
- **E2E**：每条 `pred` 携带 `frame_ts`（窗口最新帧入队时刻），前端 `Date.now() - frame_ts` 即真实画面→预测延迟

---

## API 参考

| 接口 | 说明 |
|---|---|
| `GET /health` | 服务状态、模型、设备、ONNX 提供方 |
| `GET /resolve` | yt-dlp 解析 YouTube URL → 播放直链 + 标题 |
| `GET /stream` | 直链流式代理（透传 Range 支持拖动） |
| `POST /feedback` | 人工反馈标注（`correct`/`wrong`）→ 追加写 `logs/feedback.jsonl` |
| `WS /ws/youtube` | 双轨流式推理：`start`/`stop` 控制，1Hz 推送预测（带延迟时间戳） |
| `WS /ws/stream` | 摄像头推帧模式（兼容保留）：客户端推二进制帧 → 回传 JSON 预测 |

### `/ws/youtube` 协议

```json
// 客户端 -> 服务端
{"type":"start","url":"https://www.youtube.com/watch?v=...","play_url":"<可选, 已预解析的直链>","title":"<可选>"}
{"type":"stop"}

// 服务端 -> 客户端
{"type":"started","play_url":"/stream?u=...","title":"...","ext":"mp4"}
{"type":"warmup","frames":6}
{"type":"pred","pred":"hiphop","confidence":0.97,"probs":{...},
 "server_ts":1792059000.1,"frame_ts":1792058999.9}
{"type":"error","msg":"..."}
```

`pred` 消息中 `frame_ts` 为窗口最新帧的入队时刻（画面事件时刻），是端到端延迟的计算基准。

### `/feedback` 请求体

前端每条预测旁有"准确 / 有误"按钮，点击后携带该预测的上下文上报：

```json
{"label":"wrong","pred":"hiphop","confidence":0.62,"probs":{...},
 "frame_ts":1792058999.9,"video_url":"https://...","client_ts":1792059001.0}
```

服务端追加一行 JSON 到 `logs/feedback.jsonl`（含服务端收到时刻 `ts`），供后续人工复核与再训练挖掘。

---

## 模型

**3D-CNN (r3d_18)**：16 帧（约 2 秒）滑窗，`112×112`，输入形状 `(1, 3, 16, 112, 112)`。3D 卷积同时读取空间与时间——舞种本质上由**动作**区分（霹雳舞/锁舞/震感舞单帧相似，差别在"动"），因此需要帧序列而非单帧。

推理路径：ONNX Runtime CUDA（`checkpoints/dance_classifier_6class.onnx`，~10 ms/次）优先，异常自动回退 PyTorch；6 分类中 `no_dance` 直接判"非舞蹈"，低置信度（< 0.45）再以 `Unknown / No Dance` 兜底。

导出与测速：

```bash
python export_onnx_6class.py   # 导出 dance_classifier_6class.onnx 并测速
```

---

## 训练

### 数据集

- **来源**：AIST++（`aistplusplus_dataset` / `dance_dataset_categorized`）+ YouTube 真实场景负样本（`no_dance/`）
- **规模**：5 个舞种 **823 段**（15 支编舞，每舞种 3 支）+ **非舞蹈 141 段**（15 个独立源视频：烹饪/新闻/编程/航拍/汽车/街拍/吉他/开箱/水彩/猫/课堂/烘焙/修车/天气预报/园艺）
- **命名**：`g{GENRE}_s{SCENE}_c{CAM}_d{DANCE}_m{MUSIC}_ch{TAKE}`；负样本 `gND_sBM_c01_d{NN}_mND0_ch{MM}`（`dNN` = 源视频组号，源 URL 留档于 `no_dance/sources.json`）
- 同一编舞的多机位/多音乐变体共享**同一舞者、同一服装、同一灯光、连续动作**

### 防数据泄漏

**错误做法**：切 2 秒片段后 `train_test_split(shuffle=True)` 按片段划分 → 同源片段散落训练/验证两边，模型靠服装颜色、灯光、长相作弊，指标虚高无意义。

**修正**：`make_splits.py` 按**源组（舞种, 编舞/舞者 ID `d\d+`）分组三等分**，同源片段整体归入一个集合，`assert` 强制组跨集为零。产出 `splits/grouped_{train,val,test}.txt`：**train 274 / val 271 / test 278 段**，各集每类支持数均衡（每舞种每集 41–47 片），验证与测试集的舞者/源视频均为训练完全未见过。

### 训练与结果

```bash
python make_splits.py      # 生成组级防泄漏三路切分
python train_6class.py     # 6 分类训练 (5 舞种 + no_dance)
```

- Kinetics 预训练初始化，微调 15 epoch + 余弦退火；帧预缓存到 `frame_cache/`（一次性构建，训练吞吐提升约 80 倍）
- **外观增强**（仅训练集，作用于缓存帧之后不失效缓存）：随机水平翻转 / 缩放裁剪 / 颜色抖动——逼模型学动作而非记舞者服装体型，缓解风格相近舞种（breakdancing↔ballet_jazz、popping↔locking）的外观型混淆；label smoothing 0.1 压制过自信
- **最优权重选择**：每 epoch 验证 macro F1，仅存最优快照（末轮常过拟合，不可直接当最终权重）；验证 + 测试双报告双混淆矩阵

| 类别 | precision | recall | f1-score | support (test) |
|---|---|---|---|---|
| breakdancing | 0.9677 | 0.6522 | 0.7792 | 46 |
| ballet_jazz | 1.0000 | 0.8409 | 0.9136 | 44 |
| locking | 0.6027 | 0.9362 | 0.7333 | 47 |
| hiphop | 0.9286 | 0.8298 | 0.8764 | 47 |
| popping | 0.8958 | 0.9149 | 0.9053 | 47 |
| no_dance | 1.0000 | 1.0000 | 1.0000 | 47 |
| **accuracy** | | | **0.8633** | **278** |

**测试集准确率 86.3%**（测试集舞者/源视频均未在训练出现，非作弊指标）。**no_dance 两类评测均 F1 = 1.00**（n=47/47）——非舞蹈视频可稳定判"非舞蹈"。

输出物：`checkpoints/confusion_matrix_6class_{val,test}.png`（混淆矩阵）、`checkpoints/r3d_dance_6class.pth`（按 val macro F1 选取的最优权重）。

**易混舞种**：popping ↔ locking、breakdancing ↔ ballet_jazz 在单舞者小样本下有天然外观相似性。外观增强后两对混淆显著缓解：popping 召回 0.60 → 0.91、locking 0.26 → 0.94（类别加权损失 3x/1.5x/1.2x 实验均为跷跷板效应，已弃用）；breakdancing 召回 0.35 → 0.65 为残留短板，根治需补充更多舞者数据。

### 其他推理入口

```bash
python predict.py <视频路径>          # 单文件预测
python stream_pipeline.py <视频路径>  # 不落盘抽帧演示 (uniform 采样 / 滑窗)
```

---

## 项目结构

```
dance-pulse/
├── server.py                   # FastAPI + WebSocket 服务端 (双轨架构) ★
├── youtube_pipe.py             # YouTube 直链 FFmpeg 抽帧 (线程泵, 兼容任意事件循环) ★
├── stream_pipeline.py          # ffmpeg 管道 + 内存环形队列, 不落盘抽帧 ★
├── inference_engine.py         # ONNX 推理封装 (6 分类, CUDA 自动回退 CPU)
├── make_splits.py              # 组级防泄漏三路切分生成 ★
├── train_6class.py             # 6 分类训练 + 评估 (5 舞种 + no_dance) ★核心
├── add_no_dance_category.py    # 非舞蹈负样本生成 (源 URL 留档 sources.json)
├── predict.py                  # 单文件预测
├── export_onnx_6class.py       # 6 分类 ONNX 导出 + 测速
├── web/                        # 前端 SPA (Vite + hls.js): 播放器 + 延迟指标 + 实时预测
├── Dockerfile                  # 多阶段镜像: vite 构建前端 + CUDA 推理服务
├── docker-compose.yml          # 一键部署 (CPU), 挂载 ./checkpoints
├── docker-compose.gpu.yml      # GPU 覆盖层 (NVIDIA Container Toolkit)
├── getvideo.py                 # 视频下载
├── download_direct.py          # AIST++ 直接下载 (5 个舞种)
├── categorize_videos.py        # 按舞种整理到子目录
├── dataclear.py                # 数据清理
├── requirements.txt
├── checkpoints/                # 权重、混淆矩阵、PR 曲线、ONNX (Docker volume 挂载)
├── frame_cache/                # 一次性帧缓存 (682 个 .npy), 训练提速
├── dance_dataset_categorized/  # 按舞种分类的视频
├── splits/                     # 原始 txt 划分
└── aistplusplus_dataset/       # AIST++ 元数据
```

---

## 实现注意事项

- **Windows + `uvicorn --reload`**：uvicorn 会强制 `SelectorEventLoop`（不支持 `asyncio.create_subprocess_exec`），故 `youtube_pipe.py` 用 `subprocess.Popen` + 读取线程泵帧入队列，与事件循环类型解耦
- **ONNX Runtime CUDA**：`onnxruntime-gpu` 需与 torch 的 CUDA 大版本一致（1.22.0 ↔ CUDA 12.x；1.23+ 需 CUDA 13 不兼容）；Windows 下依赖 torch wheel 自带的 cublas/cudnn DLL，容器内由 ldconfig 注册
- **视频源限速**：FFmpeg 加 `-re` 按原生帧率拉流，否则 MP4 数秒解完导致队列冻结、帧时间戳老化

---

## 局限

- **数据分布**：仅在 AIST++ 录影棚舞蹈（干净背景、单人、稳定机位）上训练。真实世界视频（手机拍摄、多人、杂乱背景、超短片段）仍会给出预测，但可能高置信度出错——分布偏移，非 bug
- **舞者层面泄漏**：分组以编舞为单位；若同一舞者跳过多支编舞，理论上仍可能泄漏舞者外观特征。更强保证需按舞者 ID 分组
- **PR 曲线**在训练结束后用最终模型生成，非逐 epoch 输出

---

## 参考

- AIST++ Dataset: https://google.github.io/aistplusplus_dataset/
- r3d_18 (Kinetics-400 预训练): torchvision.models.video.r3d_18
