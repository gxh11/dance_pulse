# youtube_pipe.py
# FFmpeg 抽帧管道: 从 YouTube 直链 (HLS/MP4) 拉流, rawvideo(RGB24) 直接
# 泵到 stdout, 逐帧读进内存。硬约束: 全程零落盘, 不产生任何临时文件。
#
# 与 stream_pipeline.FFmpegFramePipe (同步/本地文件版) 对应, 但:
#   - 子进程由读取线程泵入队列, 帧通过 asyncio.to_thread 消费
#     (不依赖 asyncio.create_subprocess_exec —— 后者在 Windows 的
#      SelectorEventLoop 上不可用, 而 uvicorn --reload 恰好强制该循环)
#   - 支持带 Referer/User-Agent 请求头 (绕过 googlevideo 403)
#   - 每帧打上入队时刻 ts (time.time()), 供端到端延迟计算
import asyncio
import queue
import subprocess
import threading
import time

import numpy as np

from stream_pipeline import FFMPEG_BIN, INPUT_SIZE, RGB_CHANNELS

_EOF = None  # 队列哨兵: 拉流结束


class AsyncFramePipe:
    """异步 FFmpeg 拉流抽帧: url -> stdout rawvideo RGB24 -> 内存帧。

    fps=8: 抽帧速率 (推理轨 16 帧窗口约 2 秒上下文, 与训练分布一致)。
    """

    def __init__(self, url: str, headers: dict | None = None,
                 width: int = INPUT_SIZE, height: int = INPUT_SIZE, fps: int = 8):
        self.url = url
        self.headers = dict(headers or {})
        self.width = int(width)
        self.height = int(height)
        self.frame_bytes = self.width * self.height * RGB_CHANNELS
        self.fps = int(fps)
        self._proc: subprocess.Popen | None = None
        self._q: queue.Queue = queue.Queue(maxsize=128)
        self._reader: threading.Thread | None = None

    def _build_cmd(self) -> list[str]:
        cmd = [FFMPEG_BIN, "-v", "error"]
        if self.headers:
            # ffmpeg -headers 需要 "Key: Value\r\n" 格式
            hdr = "".join(f"{k}: {v}\r\n" for k, v in self.headers.items())
            cmd += ["-headers", hdr]
        cmd += [
            "-re",                               # 按原生帧率节流: 否则 MP4 会在数秒内被解完,
                                                 # 队列冻结导致帧时间戳老化, 端到端延迟失控
            "-i", self.url,                      # HLS m3u8 或 MP4 直链均可
            "-vf", f"fps={self.fps},scale={self.width}:{self.height}",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1",                            # 裸帧输出到 stdout, 不落盘
        ]
        return cmd

    def _pump(self):
        """读取线程: 阻塞读精确帧 -> 队列 (daemon, 子进程结束即退出)。"""
        stdout = self._proc.stdout
        try:
            while True:
                buf = b""
                while len(buf) < self.frame_bytes:
                    chunk = stdout.read(self.frame_bytes - len(buf))
                    if not chunk:
                        return                  # EOF (finally 发哨兵)
                    buf += chunk
                ts = time.time()                # 画面事件时刻 (入队时刻)
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(
                    self.height, self.width, RGB_CHANNELS)
                self._q.put((frame, ts))
        finally:
            self._q.put(_EOF)

    async def start(self):
        """启动 ffmpeg 子进程 (stderr 必须 DEVNULL, 防止管道写满死锁)。"""
        if self._proc is not None:
            return
        self._proc = subprocess.Popen(
            self._build_cmd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    async def frames(self):
        """异步生成器: 逐帧产出 (frame_u8, ts)。EOF 或子进程退出时结束。"""
        await self.start()
        while True:
            item = await asyncio.to_thread(self._q.get)
            if item is _EOF:
                break
            yield item

    async def close(self):
        """terminate -> wait -> kill 链, 兼容 Windows。"""
        proc = self._proc
        self._proc = None
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.terminate()
        except OSError:
            return
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
