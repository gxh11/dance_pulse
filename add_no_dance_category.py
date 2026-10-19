# add_no_dance_category.py — 生成非舞蹈 (gND) 训练片段
#
# 数据约定 (make_splits.py 按 dNN 组防泄漏切分):
#   一个源视频 = 一个组: gND_sBM_c01_dNN_mND0_chMM.mp4, dNN 自动递增 (不与现有冲突)
# 零完整视频落盘: 只下载 ~1 分钟的切片片段作中转, 切成 6s 训练小段后立即删除。
# 切分由 make_splits.py 统一负责 (按组三等分), 本脚本不写 splits
# (旧版追加 crossmodal_*.txt 与 d00 命名会造成组级泄漏, 已废弃)。
import glob
import json
import os
import subprocess
import sys

import imageio_ffmpeg

# yt-dlp / ffmpeg 不在 PATH: 走 python 模块与 imageio-ffmpeg 自带二进制
YTDLP = [sys.executable, "-m", "yt_dlp"]
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

DATA_DIR = "./dance_dataset_categorized/no_dance"
SEG_SECONDS = 6
CLIPS_PER_SOURCE = 9   # 每源 9 片; 15 个源组 -> train/val/test 各 ~45, 对齐各舞种 val 支持数
FRAGMENT = "*00:30-01:24"  # 跳过片头, 取 54s (~9 片)

# 新一批非舞蹈场景 (与既有 8 源: 烹饪/新闻/编程/航拍/汽车/街拍/吉他/开箱 不重复)
QUERIES = [
    "watercolor painting tutorial speedpaint",
    "cat playing at home funny moments",
    "university lecture classroom professor",
    "baking bread recipe tutorial kitchen",
    "mechanic repairing car engine workshop",
    "weather forecast news report",
    "gardening planting flowers tutorial",
]

os.makedirs(DATA_DIR, exist_ok=True)

# 现有组号 -> 新组号顺延
existing = set()
for f in glob.glob(os.path.join(DATA_DIR, "gND_*.mp4")):
    parts = os.path.basename(f).split("_")
    if len(parts) > 3 and parts[3].startswith("d") and parts[3][1:].isdigit():
        existing.add(int(parts[3][1:]))
next_d = max(existing) + 1 if existing else 1

print(f"[*] 现有组号: {sorted(existing)} -> 新组从 d{next_d:02d} 开始")
made = 0
sources_log = []  # 源 URL 留档 (泄漏审计要求: 可回溯验证源视频互异)
for i, q in enumerate(QUERIES):
    d_idx = next_d + i
    base = f"gND_sBM_c01_d{d_idx:02d}_mND0"
    frag = os.path.join(DATA_DIR, f"_frag_d{d_idx:02d}.mp4")

    # 0) 先解析出真实 URL 并留档 (之后按 URL 下载, 保证留档与实际一致)
    res = subprocess.run(
        [*YTDLP, f"ytsearch1:{q}", "--print", "webpage_url",
         "--playlist-items", "1", "--no-download", "--quiet", "--no-warnings"],
        capture_output=True, text=True)
    url = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""

    # 1) 只取 54s 切片片段 (中转, 不是完整视频)
    dl = subprocess.run([
        *YTDLP,
        url or f"ytsearch1:{q}",
        "--format", "bestvideo[ext=mp4][height<=480]/best[ext=mp4][height<=480]/best",
        "--download-sections", FRAGMENT,
        "--ffmpeg-location", FFMPEG,  # yt-dlp 的部分下载需要它能找到 ffmpeg
        "--force-overwrites",
        "--output", frag,
        "--no-playlist", "--ignore-errors", "--quiet",
    ])
    if dl.returncode != 0 or not os.path.exists(frag):
        print(f"[!] 源 {i+1}/{len(QUERIES)} 拉取失败, 跳过: {q}")
        continue

    # 2) 切成 6s 训练小段 (keyframe 对齐即可, 长度不影响训练抽帧)
    subprocess.run([
        FFMPEG, "-y", "-v", "error", "-i", frag,
        "-c", "copy", "-f", "segment", "-segment_time", str(SEG_SECONDS),
        "-reset_timestamps", "1",
        os.path.join(DATA_DIR, f"{base}_ch%02d.mp4"),
    ], check=False)
    os.remove(frag)  # 零落盘: 中转片段立即删除

    clips = sorted(glob.glob(os.path.join(DATA_DIR, f"{base}_ch*.mp4")))
    sources_log.append({"group": f"d{d_idx:02d}", "query": q, "url": url, "clips": len(clips)})
    print(f"[+] d{d_idx:02d} ({q}): {len(clips)} 片  {url}")
    made += 1

with open(os.path.join(DATA_DIR, "sources.json"), "w", encoding="utf-8") as f:
    json.dump(sources_log, f, ensure_ascii=False, indent=2)
print(f"[*] 完成: 新增 {made} 个源组, 源 URL 已留档 no_dance/sources.json。下一步运行 python make_splits.py 重新切分")
