# download_direct.py
import glob
import os
import urllib.request
from collections import Counter

# 5 类目标舞种（与 splits 目录 txt 中实际出现的类别一致）
TARGET_GENRES = {'gBR', 'gJB', 'gLO', 'gMH', 'gPO'}
OUTPUT_DIR = "./dance_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 基础 URL 模板 (AIST 官方 10M 分辨率视频源)
# 官方域名已由 ongaacc.jp 迁移至 ongaaccel.jp
BASE_URL = "https://aistdancedb.ongaaccel.jp/v1.0.0/video/10M/{}.mp4"

# 读取 splits 目录下所有 txt，取并集作为下载清单（均为 c01 机位，与官方文件名一致）
valid_seqs = set()
for split_file in sorted(glob.glob(os.path.join("splits", "*.txt"))):
    with open(split_file, "r") as f:
        for line in f:
            seq = line.strip()
            if seq and seq.split("_")[0] in TARGET_GENRES:
                valid_seqs.add(seq)

valid_seqs = sorted(valid_seqs)
genre_counts = Counter(s.split("_")[0] for s in valid_seqs)
print(f"共 {len(valid_seqs)} 个视频片段，各类别数量：{dict(sorted(genre_counts.items()))}")

# 批量拉取视频（已存在的跳过）
total = len(valid_seqs)
for idx, seq in enumerate(valid_seqs, 1):
    filename = f"{seq}.mp4"
    save_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(save_path):
        continue
    url = BASE_URL.format(seq)
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"[{idx}/{total}] 成功下载: {filename}")
    except Exception as e:
        print(f"[{idx}/{total}] 下载失败 {filename}: {e}")
