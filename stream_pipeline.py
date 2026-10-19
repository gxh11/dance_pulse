    # stream_pipeline.py
# ffmpeg 管道 + 内存环形队列, 实现"不落盘"抽帧
#
# 硬约束 (项目约定): 处理视频时不得把整段视频 / 帧写入磁盘。
# 这里用 ffmpeg 子进程把视频解码成 rawvideo 直接写到 stdout (管道),
# 逐帧读进内存里的定长环形队列 (RingBuffer), 全程零临时文件。
import os
import shutil
import sys
import subprocess
import threading
import numpy as np

# 优先系统 PATH 的 ffmpeg (发行版构建, 链接系统 TLS 库);
# imageio-ffmpeg 自带的静态二进制仅作兜底 —— 其静态 gnutls 在部分
# 容器环境 (Docker/WSL2) 拉 HTTPS 流会段错误, 本地文件则无碍。
_sys_ffmpeg = shutil.which("ffmpeg")
if _sys_ffmpeg:
    FFMPEG_BIN = _sys_ffmpeg
else:
    try:
        import imageio_ffmpeg
        FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        FFMPEG_BIN = "ffmpeg"                       # 最后回退 PATH 字面量

# 与训练 / predict.py 完全一致
INPUT_SIZE = 112
CLIP_FRAMES = 16
RGB_CHANNELS = 3
FRAME_BYTES = INPUT_SIZE * INPUT_SIZE * RGB_CHANNELS   # 每帧 rawvideo 字节数


# ---------------------------------------------------------------------------
# 1. 内存环形队列 (定长, O(1) 入队, 取最近 n 帧)
# ---------------------------------------------------------------------------
class RingBuffer:
    """定长内存环形队列, 存放已解码的帧 (uint8), 永不落盘。

    capacity: 队列最多保留多少帧 (旧帧被新帧覆盖)。
    线程安全: ffmpeg 生产线程写入 / 推理线程读取时用锁保护。
    """

    def __init__(self, capacity, frame_shape=(INPUT_SIZE, INPUT_SIZE, RGB_CHANNELS)):
        if capacity <= 0:
            raise ValueError("capacity 必须 > 0")
        self.capacity = int(capacity)
        self.frame_shape = tuple(frame_shape)
        # 一次性分配整块内存, 避免反复分配
        self._buf = np.zeros((self.capacity, *self.frame_shape), dtype=np.uint8)
        self._ts = np.full(self.capacity, np.nan, dtype=np.float64)  # 每帧入队时刻
        self._head = 0        # 下一个写入位置
        self._count = 0       # 已写入总数 (未丢弃前的计数)
        self._lock = threading.Lock()

    def __len__(self):
        with self._lock:
            return min(self._count, self.capacity)

    def append(self, frame, ts=None):
        """入队一帧 (shape 必须等于 frame_shape), 覆盖最旧的一帧。

        ts: 该帧的采集/入队时刻 (float, 秒)。用于端到端延迟打点。
        """
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.shape != self.frame_shape:
            raise ValueError(f"帧形状 {frame.shape} != 期望 {self.frame_shape}")
        with self._lock:
            self._buf[self._head] = frame
            self._ts[self._head] = np.nan if ts is None else float(ts)
            self._head = (self._head + 1) % self.capacity
            self._count += 1

    def get_window(self, n):
        """返回最近 n 帧 (按时间先后, 旧 -> 新) 的副本。

        若已有帧数 < n, 则用最早一帧在左侧补齐 (与训练里的 pad 逻辑一致)。
        """
        return self.get_window_ts(n)[0]

    def get_window_ts(self, n):
        """返回 (window, newest_ts): 最近 n 帧副本 + 最新一帧的入队时刻。

        newest_ts 用于端到端延迟打点 (画面事件时刻)。
        """
        n = int(n)
        with self._lock:
            avail = min(self._count, self.capacity)
            if avail == 0:
                raise RuntimeError("环形队列为空, 尚未收到任何帧")
            # 逻辑顺序: 最旧 -> 最新
            idx = [(self._head - k) % self.capacity for k in range(avail, 0, -1)]
            frames = self._buf[idx]                    # (avail, H, W, C) 副本
            newest_ts = float(self._ts[idx[-1]])       # 最新一帧的入队时刻
        if n <= avail:
            return frames[-n:].copy(), newest_ts
        # 不足 n 帧: 用第一帧左侧补齐
        pad = np.repeat(frames[:1], n - avail, axis=0)
        return np.concatenate([pad, frames], axis=0), newest_ts

    def clear(self):
        with self._lock:
            self._head = 0
            self._count = 0


# ---------------------------------------------------------------------------
# 2. ffmpeg 管道 (子进程 stdout -> 内存, 不落盘)
# ---------------------------------------------------------------------------
class FFmpegFramePipe:
    """用 ffmpeg 解码视频, 把 rawvideo(RGB24) 直接泵到 stdout, 逐帧读进内存。

    全程不产生任何临时文件; ffmpeg 的 stderr 被丢弃 (设 capture_stderr=True 可留存)。
    """

    def __init__(self, video_path, width=INPUT_SIZE, height=INPUT_SIZE, capture_stderr=False):
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)
        self.video_path = video_path
        self.width = int(width)
        self.height = int(height)
        self.frame_bytes = self.width * self.height * RGB_CHANNELS

        cmd = [
            FFMPEG_BIN,
            "-v", "error",          # 只在出错时打印, 保持安静
            "-i", video_path,
            "-f", "rawvideo",       # 裸帧输出
            "-pix_fmt", "rgb24",    # RGB, 每通道 8 bit
            "-vf", f"scale={self.width}:{self.height}",
            "-an",                  # 丢弃音频
            "-"                     # 输出到 stdout
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
            bufsize=self.frame_bytes * 4,   # 大缓冲, 减少系统调用
        )

    def read_frame(self):
        """从管道精确读取一帧; 到达 EOF 或解码失败返回 None。"""
        buf = self._proc.stdout.read(self.frame_bytes)
        if not buf or len(buf) < self.frame_bytes:
            return None
        return np.frombuffer(buf, dtype=np.uint8).reshape(
            self.height, self.width, RGB_CHANNELS)   # (H, W, C)

    def stream(self):
        """生成器: 逐帧产出 (index, frame), 全程内存中。"""
        i = 0
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            yield i, frame
            i += 1

    def close(self):
        if self._proc.poll() is None:
            self._proc.stdout.close()
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 3. 便捷函数: 流式抽帧 -> 模型输入 (与训练同款归一化)
# ---------------------------------------------------------------------------
def normalize_clip(clip):
    """(T, H, W, C) uint8 -> (C, T, H, W) float32, 与训练/predict.py 完全一致。"""
    arr = clip.astype(np.float32) / 255.0
    arr = (arr - np.float32([0.45, 0.45, 0.45])) / np.float32([0.225, 0.225, 0.225])
    return (arr.astype(np.float32)).transpose(3, 0, 1, 2)


def stream_sample_clip(video_path, num_frames=CLIP_FRAMES,
                       width=INPUT_SIZE, height=INPUT_SIZE):
    """用 ffmpeg 管道流式均匀采样 num_frames 帧, 不落盘。

    策略: 顺序解码, 用蓄水池思想只把落在均匀网格上的帧留进内存,
    其余帧读出后立即丢弃 (不写盘、不缓存整段)。
    """
    import cv2  # 仅用于取总帧数(不负责解码), 解码走 ffmpeg 管道
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    total = max(total, 1)

    want = np.linspace(0, total - 1, num_frames, dtype=int)  # 目标帧号
    want_set = {}
    for k, fi in enumerate(want):
        want_set.setdefault(int(fi), k)

    picked = [None] * num_frames
    with FFmpegFramePipe(video_path, width, height) as pipe:
        for i, frame in pipe.stream():
            if i in want_set:
                picked[want_set[i]] = frame
            if i >= want[-1]:
                break

    # 解码到 EOF 仍缺帧(帧数估计偏大) -> 用最后一帧补齐
    last = next((f for f in reversed(picked) if f is not None),
                np.zeros((height, width, RGB_CHANNELS), dtype=np.uint8))
    for k in range(num_frames):
        if picked[k] is None:
            picked[k] = last
    return np.stack(picked)   # (T, H, W, C) uint8


def ring_inference_window(video_path, ring_capacity=32):
    """演示用: 逐帧流入环形队列, 取最近 CLIP_FRAMES 帧作为模型输入。

    这是"实时滑窗"用法 —— 队列只保留最近 ring_capacity 帧, 旧帧被覆盖,
    任意时刻取最近 16 帧即可推理, 全程内存、零落盘。
    """
    ring = RingBuffer(capacity=ring_capacity)
    with FFmpegFramePipe(video_path) as pipe:
        for _, frame in pipe.stream():
            ring.append(frame)          # O(1) 入队, 覆盖最旧帧
            if len(ring) >= CLIP_FRAMES:
                window = ring.get_window(CLIP_FRAMES)   # (16, H, W, C)
                # 在实际推理中: model(normalize_clip(window)) ...
    return ring


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("用法: python stream_pipeline.py <视频路径>")
    video = sys.argv[1]
    if not os.path.exists(video):
        raise SystemExit(f"[!] 找不到视频: {video}")

    # --- 模式 A: 一次性流式均匀采样 16 帧 (不落盘) ---
    clip = stream_sample_clip(video)
    print(f"[A] 流式抽帧 -> clip {clip.shape}, dtype={clip.dtype}")
    print(f"    归一化后模型输入: {normalize_clip(clip).shape} (C,T,H,W)")

    # --- 模式 B: 环形队列滑窗, 逐帧流入, 取最近 16 帧 ---
    ring = RingBuffer(capacity=32)
    n = 0
    with FFmpegFramePipe(video) as pipe:
        for _, frame in pipe.stream():
            ring.append(frame)
            n += 1
    window = ring.get_window(CLIP_FRAMES)
    print(f"[B] 环形队列: 共流入 {n} 帧, 队列容量 {ring.capacity}")
    print(f"    最近 {CLIP_FRAMES} 帧窗口: {window.shape}, dtype={window.dtype}")
    print("[*] 全程内存管道 + 环形队列, 未写入任何帧/临时文件 ✔")
