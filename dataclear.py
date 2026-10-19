import os
import glob
from collections import defaultdict

# ==================== 1. 配置规则 ====================
# 目标5大舞种: gBR(Breakdance), gPO(Popping), gLO(Locking), gMH(Hip-Hop), gJB(Ballet Jazz)
TARGET_GENRES = {"gBR", "gPO", "gLO", "gMH", "gJB"}

# 是否只保留 c01 正面单机位 (强烈建议为 True，防止9机位冗余)
KEEP_C01_ONLY = True

# 扫描的目标后缀 (包含关节点、运动参数、特征与视频文件)
TARGET_EXTS = {".pkl", ".json", ".mp4", ".npy", ".csv"}

# ==================== 2. 加载并规范化黑名单 ====================
ignore_set = set()
ignore_base_set = set()

if os.path.exists("ignore_list.txt"):
    with open("ignore_list.txt", "r") as f:
        for line in f:
            item = line.strip()
            if not item:
                continue
            ignore_set.add(item)
            parts = item.split("_")
            if len(parts) >= 6:
                # 提取无机位基础标识: gXX_sXX_dXX_mXX_chXX
                base_id = f"{parts[0]}_{parts[1]}_{parts[3]}_{parts[4]}_{parts[5]}"
                ignore_base_set.add(base_id)
    print(f"[*] 已加载黑名单 ignore_list.txt，共 {len(ignore_set)} 条记录。")
else:
    print("[!] 未检测到 ignore_list.txt，将仅根据舞种与机位规则进行清洗。")

# ==================== 3. 物理扫描与删除无效数据文件 ====================
total_scanned = 0
deleted_genre = 0
deleted_ignore = 0
deleted_camera = 0
retained_by_genre = defaultdict(int)

for root, _, files in os.walk("."):
    # 忽略 git 与脚本自身
    if ".git" in root:
        continue
    for file in files:
        name, ext = os.path.splitext(file)
        if ext.lower() not in TARGET_EXTS:
            continue
        
        total_scanned += 1
        file_path = os.path.join(root, file)
        parts = name.split("_")
        
        # AIST++ 标准命名格式至少包含 6 段 (gXX_sXX_cXX_dXX_mXX_chXX)
        if len(parts) < 6:
            continue

        genre = parts[0]
        cam = parts[2]
        base_id = f"{parts[0]}_{parts[1]}_{parts[3]}_{parts[4]}_{parts[5]}"

        # 规则 1: 舞种过滤
        if genre not in TARGET_GENRES:
            os.remove(file_path)
            deleted_genre += 1
            continue

        # 规则 2: 黑名单过滤
        if name in ignore_set or base_id in ignore_base_set:
            os.remove(file_path)
            deleted_ignore += 1
            continue

        # 规则 3: 机位过滤 (仅保留 c01)
        if KEEP_C01_ONLY and cam != "c01":
            os.remove(file_path)
            deleted_camera += 1
            continue

        # 符合要求，予以保留
        retained_by_genre[genre] += 1

# ==================== 4. 清洗 splits 划分清单 ====================
split_files = glob.glob("splits/**/*.txt", recursive=True) + glob.glob("splits/*.txt")
for sp_path in split_files:
    with open(sp_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    cleaned_lines = []
    for seq in lines:
        parts = seq.split("_")
        if len(parts) < 6:
            continue
        genre = parts[0]
        base_id = f"{parts[0]}_{parts[1]}_{parts[3]}_{parts[4]}_{parts[5]}"
        
        if genre in TARGET_GENRES and seq not in ignore_set and base_id not in ignore_base_set:
            if KEEP_C01_ONLY:
                seq = f"{parts[0]}_{parts[1]}_c01_{parts[3]}_{parts[4]}_{parts[5]}"
            cleaned_lines.append(seq)
            
    # 去重并覆写
    cleaned_lines = sorted(list(set(cleaned_lines)))
    with open(sp_path, "w") as f:
        f.write("\n".join(cleaned_lines) + "\n")
    print(f"[*] 已同步重写划分文件: {sp_path} (剩余有效条目: {len(cleaned_lines)})")

# ==================== 5. 输出清洗总结报告 ====================
print("\n" + "=" * 45)
print("              数据物理清洗完成报告")
print("=" * 45)
print(f"扫描数据文件总数: {total_scanned}")
print(f"删除非目标舞种文件: {deleted_genre}")
print(f"删除黑名单异常文件: {deleted_ignore}")
if KEEP_C01_ONLY:
    print(f"删除冗余机位(c02-c09): {deleted_camera}")
total_deleted = deleted_genre + deleted_ignore + deleted_camera
print(f"物理删除文件总计: {total_deleted}")
print("-" * 45)
print("各舞种保留文件数量统计 (仅 c01 视角):")
for g, count in sorted(retained_by_genre.items()):
    print(f"  - {g}: {count} 个文件")
print("=" * 45)