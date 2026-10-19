# categorize_videos.py
import os
import shutil

SOURCE_VIDEO_DIR = "./dance_videos"
OUTPUT_BASE_DIR = "./dance_dataset_categorized"

# 严格只保留选定的 5 种舞种
GENRE_MAP = {
    "gBR": "breakdancing",
    "gJB": "ballet_jazz",
    "gLO": "locking",
    "gMH": "hiphop",
    "gPO": "popping"
}

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
stats = {name: 0 for name in GENRE_MAP.values()}

for filename in os.listdir(SOURCE_VIDEO_DIR):
    if not filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        continue

    genre_code = filename.split("_")[0]
    if genre_code in GENRE_MAP:
        class_name = GENRE_MAP[genre_code]
        target_dir = os.path.join(OUTPUT_BASE_DIR, class_name)
        os.makedirs(target_dir, exist_ok=True)
        
        src_file = os.path.join(SOURCE_VIDEO_DIR, filename)
        dst_file = os.path.join(target_dir, filename)
        shutil.move(src_file, dst_file)
        stats[class_name] += 1

print("归类完成：")
for k, v in stats.items():
    print(f"  - {k}: {v} 个视频")