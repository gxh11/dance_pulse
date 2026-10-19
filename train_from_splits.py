# train_from_splits.py
import os
import re
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve, average_precision_score)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from multiprocessing import Pool

# 1. 舞种映射 (5 类别)
GENRE_MAP = {
    "gBR": 0,  # breakdancing
    "gJB": 1,  # ballet_jazz
    "gLO": 2,  # locking
    "gMH": 3,  # hiphop
    "gPO": 4   # popping
}
CLASS_NAMES = ["breakdancing", "ballet_jazz", "locking", "hiphop", "popping"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-4
CLIP_FRAMES = 16  # 2 秒上下文 (8 fps)
INPUT_SIZE = 112
CACHE_DIR = "./frame_cache"  # 一次性帧缓存, 避免每个 epoch 重复解码视频
SEED = 42
VAL_DANCES_PER_GENRE = 1  # 每个舞种抽 1 支"编舞/源视频"作验证(完全未见过), 其余作训练

# 文件名形如 gBR_sBM_c01_d04_mBR0_ch01, 其中 d\d+ 是"编舞/源表演" ID
DANCE_RE = re.compile(r'^d\d+$')

def cache_path(vpath):
    base = os.path.splitext(os.path.basename(vpath))[0]
    return os.path.join(CACHE_DIR, base + ".npy")

def extract_clip(vpath, num_frames=CLIP_FRAMES):
    """直接 seek 采样 16 帧, 而不是顺序解码整个视频"""
    cap = cv2.VideoCapture(vpath)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, max(0, total - 1), num_frames, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE)), cv2.COLOR_BGR2RGB))
    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8))
    return np.stack(frames)  # (T, H, W, C) uint8

def build_cache_one(vpath):
    np.save(cache_path(vpath), extract_clip(vpath))

# 2. 按"源视频/编舞"分组的数据集 —— 同源样本绝不跨集
class VideoDataset(Dataset):
    def __init__(self, samples, num_frames=CLIP_FRAMES):
        # samples: list of (vpath, label, group_key)
        self.samples = samples
        self.num_frames = num_frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vpath, label, _ = self.samples[idx]
        cpath = cache_path(vpath)
        if os.path.exists(cpath):
            clip = np.load(cpath)  # (T, H, W, C) uint8
        else:
            clip = extract_clip(vpath, self.num_frames)

        # (T, H, W, C) -> (C, T, H, W) 并归一化 (必须保持 float32, 否则 numpy 会提升为 float64)
        arr = clip.astype(np.float32) / 255.0
        arr = (arr - np.float32([0.45, 0.45, 0.45])) / np.float32([0.225, 0.225, 0.225])
        tensor = torch.from_numpy(arr.astype(np.float32)).permute(3, 0, 1, 2)
        return tensor, torch.tensor(label, dtype=torch.long)

def parse_group(seq):
    """返回"源视频/编舞"分组键 (genre, dance_id)。

    同一支编舞的所有机位/音乐/片段 (如 d04 的 ch01..ch10) 都属于同一组,
    这样它们会被整体分到训练或验证, 绝不跨集 —— 从而杜绝
    "同舞者 / 同服装 / 同灯光 / 连续动作" 从切片层面泄漏。"""
    parts = seq.split('_')
    genre = parts[0]
    dance = next((p for p in parts if DANCE_RE.match(p)), None)
    return (genre, dance)

def scan_videos(search_dirs):
    """扫描磁盘视频 -> list of (path, seq_name)"""
    out = []
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for root, _, files in os.walk(d):
            for f in sorted(files):
                if f.lower().endswith(('.mp4', '.avi', '.mov')):
                    out.append((os.path.join(root, f), os.path.splitext(f)[0]))
    return out

def build_grouped_split(search_dirs, val_per_genre=VAL_DANCES_PER_GENRE, seed=SEED):
    """按"源视频/编舞"分组, 再按舞种分层划分 train/val。

    关键: 以"组"为单位整体分配, 绝不按切片/单文件随机划分。
    同源的所有片段要么全进训练, 要么全进验证, 从根上消除数据泄漏。"""
    videos = scan_videos(search_dirs)
    groups = {}  # group_key -> list of (path, label, group_key)
    for path, seq in videos:
        genre = seq.split('_')[0]
        if genre not in GENRE_MAP:
            continue
        gid = parse_group(seq)
        if gid[1] is None:
            continue
        groups.setdefault(gid, []).append((path, GENRE_MAP[genre], gid))

    rng = random.Random(seed)
    train, val = [], []
    for genre in sorted({g for g, _ in groups}):
        gkeys = sorted([k for k in groups if k[0] == genre])
        rng.shuffle(gkeys)               # 打乱该舞种下的编舞分组
        val_keys = gkeys[:val_per_genre]  # 抽 N 支编舞整体作验证
        train_keys = gkeys[val_per_genre:]
        for k in train_keys:
            train.extend(groups[k])
        for k in val_keys:
            val.extend(groups[k])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val

if __name__ == "__main__":
    # 3. 按"源视频/编舞"做防泄漏分组划分 (不再用会跨集泄漏的 crossmodal txt)
    SEARCH_DIRS = ["./dance_dataset_categorized"]
    train_samples, val_samples = build_grouped_split(SEARCH_DIRS)

    # 打印划分情况, 便于确认没有任何编舞跨集
    def _dances(samples):
        return sorted({g for _, _, g in samples})
    print(f"[*] 训练样本: {len(train_samples)} | 验证样本: {len(val_samples)}")
    print(f"[*] 训练编舞: {len(_dances(train_samples))} 支 | 验证编舞: {len(_dances(val_samples))} 支")
    assert not (set(_dances(train_samples)) & set(_dances(val_samples))), \
        "数据泄漏: 有编舞同时出现在训练与验证!"

    train_set = VideoDataset(train_samples, num_frames=CLIP_FRAMES)
    val_set = VideoDataset(val_samples, num_frames=CLIP_FRAMES)

    # 一次性构建帧缓存 (仅缺失的视频, 并行提取)
    os.makedirs(CACHE_DIR, exist_ok=True)
    all_videos = list(dict.fromkeys(vp for vp, _, _ in train_samples + val_samples))
    missing = [vp for vp in all_videos if not os.path.exists(cache_path(vp))]
    if missing:
        print(f"[*] 构建帧缓存: {len(missing)}/{len(all_videos)} 个视频待提取 ...")
        with Pool(processes=min(10, os.cpu_count() or 4)) as pool:
            list(tqdm(pool.imap_unordered(build_cache_one, missing), total=len(missing),
                      desc="Cache build", unit="video"))
    else:
        print(f"[*] 帧缓存已就绪: {len(all_videos)} 个视频")

    NUM_WORKERS = 4
    # 注意: 不使用 persistent_workers —— 它会在 epoch 边界触发 CUDA 释放崩溃 (c10 abort)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    # 4. 初始化 3D-CNN (r3d_18)
    print("[*] 正在加载 torchvision 预训练 r3d_18 (3D-CNN)...")
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

    # 5. 模型微调
    os.makedirs("./checkpoints", exist_ok=True)
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", unit="batch")
        for videos, labels in pbar:
            videos, labels = videos.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{correct*100/total:.2f}%")

        acc = correct / total if total > 0 else 0
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/max(total,1):.4f} - Train Acc: {acc*100:.2f}%")

        # 每个 epoch 都保存权重, 崩溃最多损失 1 个 epoch
        torch.save(model.state_dict(), "./checkpoints/r3d_dance_best.pth")
        torch.save(model.state_dict(), f"./checkpoints/r3d_epoch{epoch+1}.pth")
        print(f"[*] 已保存 checkpoint (epoch {epoch+1})")
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # 6. 在"完全未见过的编舞"验证集上评估
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for videos, labels in tqdm(val_loader, desc="Validation", unit="batch"):
            videos, labels = videos.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            outputs = model(videos)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n" + "="*50)
    print("   防泄漏验证集评估 (按源视频/编舞分组, 未见过的编舞)")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4, zero_division=0))

    # 绘制并保存混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Validation Confusion Matrix (Grouped by source video / no leakage)')
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.tight_layout()
    plt.savefig('./checkpoints/confusion_matrix.png')
    print("[*] 混淆矩阵图已保存至: ./checkpoints/confusion_matrix.png")

    # 绘制并保存 Precision-Recall 曲线 (每类 + micro 平均)
    y_true = np.eye(len(CLASS_NAMES))[np.asarray(all_labels)]   # one-hot
    y_score = np.asarray(all_probs)
    plt.figure(figsize=(8, 6))
    aps = []
    for i, name in enumerate(CLASS_NAMES):
        p, r, _ = precision_recall_curve(y_true[:, i], y_score[:, i])
        ap = average_precision_score(y_true[:, i], y_score[:, i])
        aps.append(ap)
        plt.plot(r, p, lw=2, label=f"{name} (AP={ap:.3f})")
    p_micro, r_micro, _ = precision_recall_curve(y_true.ravel(), y_score.ravel())
    ap_micro = average_precision_score(y_true, y_score, average="micro")
    plt.plot(r_micro, p_micro, lw=2.5, ls="--", color="black",
             label=f"micro-average (AP={ap_micro:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.title("Precision-Recall Curves (Grouped split / no leakage)")
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('./checkpoints/pr_curve.png')
    print(f"[*] PR 曲线已保存至: ./checkpoints/pr_curve.png  (micro AP={ap_micro:.4f})")
