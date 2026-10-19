# add_dancer_diversity.py — 为易混舞种补充"新舞者"源组 (修复 breakdancing<->ballet_jazz / popping<->locking 混淆)
#
# 背景: 每舞种只有 3 个源组 (1 舞者/编舞), 三等分后训练只见过 1 个舞者 ->
# 模型靠服装/体型/背景记忆而非动作特征, 风格相近的舞种在"没见过的舞者"上互相误判。
# 根治 = 增加舞者多样性。本脚本从 YouTube 拉取真实舞者片段, 作为新的源组入库。
#
# 数据约定 (make_splits.py 按 dNN 组防泄漏切分):
#   一个源视频 = 一个组: g{GEN}_sYT_c01_dNN_mYT0_chMM.mp4
#   dNN 从 40 起编号, 与 AIST++ (d04-24) / gND (d01-15) 命名空间不碰撞。
# 零完整视频落盘: 只下载 ~54s 中转切片, 切成 6s 训练小段后立即删除。
# 源 URL 留档 {genre}/sources_yt.json (泄漏审计: 可回溯验证源视频互异)。
import glob
import json
import os
import subprocess
import sys

import imageio_ffmpeg

# yt-dlp / ffmpeg 不在 PATH: 走 python 模块与 imageio-ffmpeg 自带二进制
YTDLP = [sys.executable, "-m", "yt_dlp"]
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

ROOT = "./dance_dataset_categorized"
SEG_SECONDS = 6
CLIPS_PER_SOURCE = 9   # 每源 9 片 (54s)
FRAGMENT = "*00:30-01:24"        # 跳过片头, 取 54s (~9 片)
FRAGMENT_FALLBACK = "*00:10-01:04"  # 短视频回退窗口

# 每舞种 3 个新源 (不同舞者/场景), 查询词避开 AIST++ 录影棚风格以增加多样性
TARGETS = {
    "gBR": [  # breakdancing — 与 ballet_jazz 地面动作混淆
        "breakdance solo top rock footwork performance",
        "breaking dance battle bboy solo",
        "breakdance powermove routine showcase",
    ],
    "gJB": [  # ballet_jazz — 与 breakdancing 混淆
        "jazz dance routine studio performance",
        "ballet jazz dance choreography solo",
        "contemporary jazz dance performance stage",
    ],
    "gPO": [  # popping — 与 locking 混淆
        "popping dance solo performance",
        "popping animation dance showcase",
        "boogaloo popping dance routine",
    ],
    "gLO": [  # locking — 与 popping 混淆
        "locking dance solo performance",
        "locking funk dance showcase",
        "lock dance routine solo",
    ],
}
D_START = 40  # 新组 dNN 起始号 (不与 AIST++/gND 碰撞)


def download_group(genre: str, d_idx: int, query: str, used_urls: set):
    """返回 (片段数, 源 URL); URL 用于 sources_yt.json 留档。"""
    base = f"{genre}_sYT_c01_d{d_idx:02d}_mYT0"
    data_dir = os.path.join(ROOT, genre)
    os.makedirs(data_dir, exist_ok=True)
    frag = os.path.join(data_dir, f"_frag_d{d_idx:02d}.mp4")

    # 0) 先解析出真实 URL 并留档 (之后按 URL 下载, 保证留档与实际一致)
    res = subprocess.run(
        [*YTDLP, f"ytsearch1:{query}", "--print", "webpage_url",
         "--playlist-items", "1", "--no-download", "--quiet", "--no-warnings"],
        capture_output=True, text=True)
    url = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""
    if not url:
        print(f"[!] 解析 URL 失败: {query}")
        return 0, ""
    if url in used_urls:
        print(f"[!] 重复源 URL, 跳过 (同舞者/同视频不能当两个组): {query} -> {url}")
        return 0, url

    # 1) 只取 54s 切片片段 (中转, 不是完整视频)
    def try_dl(fragment):
        return subprocess.run([
            *YTDLP, url,
            "--format", "bestvideo[ext=mp4][height<=480]/best[ext=mp4][height<=480]/best",
            "--download-sections", fragment,
            "--ffmpeg-location", FFMPEG,  # yt-dlp 的部分下载需要它能找到 ffmpeg
            "--force-overwrites",
            "--output", frag,
            "--no-playlist", "--ignore-errors", "--quiet",
        ])

    def frag_ok() -> bool:
        """片段必须含视频流且非空 (曾出现 0 流片段 -> 切分报 'does not contain any stream')"""
        if not os.path.exists(frag) or os.path.getsize(frag) == 0:
            return False
        probe = subprocess.run(
            [FFMPEG, "-i", frag], capture_output=True, text=True)  # -i 无输出文件时 rc!=0 但 stderr 含流信息
        return "Video:" in probe.stderr

    try_dl(FRAGMENT)
    if not frag_ok():
        try_dl(FRAGMENT_FALLBACK)  # 短视频/空片段回退
    if not frag_ok():
        print(f"[!] 拉取失败或片段无视频流, 跳过: {query}")
        if os.path.exists(frag):
            os.remove(frag)
        return 0, url

    # 2) 切成 6s 训练小段 (keyframe 对齐即可, 长度不影响训练抽帧)
    subprocess.run([
        FFMPEG, "-y", "-v", "error", "-i", frag,
        "-c", "copy", "-f", "segment", "-segment_time", str(SEG_SECONDS),
        "-reset_timestamps", "1",
        os.path.join(data_dir, f"{base}_ch%02d.mp4"),
    ], check=False)
    if os.path.exists(frag):
        os.remove(frag)  # 零落盘: 中转片段立即删除

    clips = sorted(glob.glob(os.path.join(data_dir, f"{base}_ch*.mp4")))
    used_urls.add(url)
    print(f"[+] {genre} d{d_idx:02d} ({query}): {len(clips)} 片  {url}")
    return len(clips), url


def main():
    used_urls: set = set()
    # 已留档的 URL 也视为已用 (避免重复运行时同一源生成两个组)
    for genre in TARGETS:
        p = os.path.join(ROOT, genre, "sources_yt.json")
        if os.path.exists(p):
            try:
                for e in json.load(open(p, encoding="utf-8")):
                    if e.get("url"):
                        used_urls.add(e["url"])
            except Exception:
                pass

    log_all = {}
    d_idx = D_START
    for genre, queries in TARGETS.items():
        entries = []
        for q in queries:
            n, url = download_group(genre, d_idx, q, used_urls)
            entries.append({"group": f"d{d_idx:02d}", "query": q, "clips": n, "url": url})
            d_idx += 1
        log_all[genre] = entries
        with open(os.path.join(ROOT, genre, "sources_yt.json"), "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

    total = sum(e["clips"] for es in log_all.values() for e in es)
    print(f"[*] 完成: 新增 {total} 片, 源 URL 留档 {ROOT}/<genre>/sources_yt.json")
    print("[*] 下一步: python make_splits.py 重新切分 (组级防泄漏)")


if __name__ == "__main__":
    main()
