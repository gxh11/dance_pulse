# train_6class.py — 6 分类训练 (5 舞种 + no_dance 非舞蹈)
#
# 关键点:
#   1. 帧缓存: frame_cache/{seq}.npy 预解码 (16,112,112,3) uint8, 命中即免视频解码
#      (相比每 epoch 重新解码视频 ~80x 提速, GPU 不再空等数据)
#   2. batch 级进度条 (tqdm), 不再"整个 epoch 无输出像卡死"
#   3. 全流程包在 if __name__ == "__main__" 内 (Windows 多进程安全)
#   4. 每个 epoch 都存 checkpoint, 崩溃不丢进度
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 1. 6 分类字典映射 (gND = 非舞蹈负样本, 命名 gND_sBM_c01_dNN_mND0_chMM)
GENRE_MAP = {
    "gBR": 0,  # breakdancing
    "gJB": 1,  # ballet_jazz
    "gLO": 2,  # locking
    "gMH": 3,  # hiphop
    "gPO": 4,  # popping
    "gND": 5,  # no_dance
}
CLASS_NAMES = ["breakdancing", "ballet_jazz", "locking", "hiphop", "popping", "no_dance"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 15
LR = 1e-4
CLIP_FRAMES = 16
INPUT_SIZE = 112
CACHE_DIR = "./frame_cache"  # 预解码帧缓存 (与 train_from_splits.py 同约定)


def cache_path(vpath: str) -> str:
    base = os.path.splitext(os.path.basename(vpath))[0]
    return os.path.join(CACHE_DIR, base + ".npy")


def extract_clip(vpath: str, num_frames: int = CLIP_FRAMES) -> np.ndarray:
    """seek 采样 16 帧 (不顺序解码整个视频) -> (T, H, W, C) uint8"""
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
    return np.stack(frames)


def augment(clip: np.ndarray) -> np.ndarray:
    """训练期外观增强 (作用于缓存帧之后, 缓存仍存原始帧, 不失效):
    逼模型学动作而非记住舞者的服装/体型/机位 —— 针对
    breakdancing<->ballet_jazz / popping<->locking 的外观型混淆。"""
    # 随机水平翻转 (镜像不改舞种)
    if np.random.rand() < 0.5:
        clip = clip[:, :, ::-1, :].copy()
    # 随机缩放裁剪 (尺度/取景抖动, 弱化机位构图记忆)
    if np.random.rand() < 0.7:
        _, H, W, _ = clip.shape
        scale = np.random.uniform(0.8, 1.0)
        ch, cw = int(H * scale), int(W * scale)
        y = np.random.randint(0, H - ch + 1)
        x = np.random.randint(0, W - cw + 1)
        clip = np.stack([cv2.resize(f, (W, H)) for f in clip[:, y:y + ch, x:x + cw, :]])
    # 颜色抖动 (亮度/对比度/饱和度, 弱化服装颜色记忆)
    if np.random.rand() < 0.8:
        clip = clip.astype(np.float32)
        clip *= np.random.uniform(0.7, 1.3)                    # 亮度
        clip = (clip - clip.mean()) * np.random.uniform(0.7, 1.3) + clip.mean()  # 对比度
        if np.random.rand() < 0.5:
            gray = clip.mean(axis=-1, keepdims=True)
            clip = gray + (clip - gray) * np.random.uniform(0.5, 1.5)  # 饱和度
        clip = np.clip(clip, 0, 255).astype(np.uint8)
    return clip


class AIST6ClassDataset(Dataset):
    def __init__(self, split_txt_path, search_dir="./dance_dataset_categorized", num_frames=CLIP_FRAMES, train=False):
        self.num_frames = num_frames
        self.train = train
        self.samples = []

        disk_videos = {}
        for root, _, files in os.walk(search_dir):
            for f in files:
                if f.lower().endswith((".mp4", ".avi", ".mov")):
                    base = os.path.splitext(f)[0]
                    disk_videos[base] = os.path.join(root, f)

        with open(split_txt_path, "r", encoding="utf-8") as f:
            lines = [line.strip().replace(".mp4", "") for line in f if line.strip()]

        missing = 0
        for seq in lines:
            genre_code = seq.split("_")[0]
            if genre_code in GENRE_MAP:
                # 严格按 seq 名匹配, 不做相机号回退 (回退可能让两行映射到同一文件 -> 同文件跨集泄漏隐患)
                matched_path = disk_videos.get(seq)
                if matched_path:
                    self.samples.append((matched_path, GENRE_MAP[genre_code]))
                else:
                    missing += 1

        counts = [0] * len(CLASS_NAMES)
        for _, lb in self.samples:
            counts[lb] += 1
        print(f"[*] {split_txt_path}: 载入 {len(self.samples)} 样本 (缺失 {missing})")
        print("    " + "  ".join(f"{n}={c}" for n, c in zip(CLASS_NAMES, counts)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vpath, label = self.samples[idx]
        cpath = cache_path(vpath)
        if os.path.exists(cpath):
            clip = np.load(cpath)  # (T, H, W, C) uint8 —— 免解码快路径
        else:
            clip = extract_clip(vpath, self.num_frames)
            os.makedirs(CACHE_DIR, exist_ok=True)
            np.save(cpath, clip)  # 顺手建缓存, 下个 epoch 起走快路径

        if self.train:
            clip = augment(clip)  # 缓存存原始帧, 每 epoch 在线增强 (不过缓存)

        # (T, H, W, C) -> (C, T, H, W) 归一化 (保持 float32, 防 numpy 提升为 float64)
        arr = clip.astype(np.float32) / 255.0
        arr = (arr - np.float32([0.45, 0.45, 0.45])) / np.float32([0.225, 0.225, 0.225])
        tensor = torch.from_numpy(arr.astype(np.float32)).permute(3, 0, 1, 2)
        return tensor, torch.tensor(label, dtype=torch.long)


def main():
    # 使用按"舞者(源视频)"分组的三路切分 (make_splits.py 生成, 组级防泄漏):
    #   旧 crossmodal_*.txt 按音乐切分, 同一舞者跨 train/val (身份泄漏), 已弃用
    train_set = AIST6ClassDataset("splits/grouped_train.txt", train=True)  # train=True 开启外观增强
    val_set = AIST6ClassDataset("splits/grouped_val.txt")
    test_set = AIST6ClassDataset("splits/grouped_test.txt")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # 3. 初始化 6 输出分类模型 (Kinetics 预训练起步)
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model = model.to(DEVICE)

    # 均匀权重。locking 加权实验已全面证伪: 3x 吞 popping (recall 0.15), 1.5x 吞 hiphop (0.13),
    # 1.2x 吞 popping (0.02) 且 locking 也没拉回 (0.11) —— 单舞者小样本下加权只是跷跷板+大方差,
    # w=1.0 + 外观增强是唯一稳定配置 (test acc 86.7%)
    class_weights = torch.ones(len(CLASS_NAMES), device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)  # 抑制对相似舞种的过自信误判
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 4. 训练: 每 epoch 存档 + 验证 macro F1 选最优 (最后一轮往往过拟合, 不能直接当最终权重)
    os.makedirs("./checkpoints", exist_ok=True)
    best_f1, best_path = -1.0, "./checkpoints/r3d_dance_6class.pth"

    def macro_f1(loader):
        from sklearn.metrics import f1_score
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for videos, labels in loader:
                videos = videos.to(DEVICE)
                outputs = model(videos)
                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                all_labels.extend(labels.numpy())
        return f1_score(all_labels, all_preds, average="macro", zero_division=0)

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for videos, labels in pbar:
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{running_loss/max(total,1):.4f}",
                             acc=f"{correct/max(total,1)*100:.1f}%")
        scheduler.step()

        acc = correct / total if total > 0 else 0
        val_f1 = macro_f1(val_loader)
        marker = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            marker = "  <- best, saved"
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss/max(total,1):.4f}"
              f" - Acc: {acc*100:.2f}% - val macro F1: {val_f1:.4f}{marker}")
        torch.save(model.state_dict(), "./checkpoints/r3d_dance_6class_last.pth")

    # 加载最优权重 (非最后一轮) 做最终评测
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    print(f"[*] 最优权重 val macro F1={best_f1:.4f}, 已加载 {best_path} 做最终评测")

    # 5. 验证集 + 测试集全量指标评测 (两集合的舞者/源视频均未在训练出现)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()

    def evaluate(loader, name):
        all_preds, all_labels = [], []
        with torch.no_grad():
            for videos, labels in tqdm(loader, desc=f"{name}评测", unit="batch"):
                videos, labels = videos.to(DEVICE), labels.to(DEVICE)
                outputs = model(videos)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        print("\n" + "=" * 50)
        print(f"       6分类{name}指标评估报告 (含 No Dance)")
        print("=" * 50)
        print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4, zero_division=0))

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title(f"{name} Confusion Matrix (6 Classes including No Dance)")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.tight_layout()
        out = f"./checkpoints/confusion_matrix_6class_{name}.png"
        plt.savefig(out)
        plt.close()
        print(f"[*] 混淆矩阵已保存至: {out}")

    evaluate(val_loader, "val")
    evaluate(test_loader, "test")


if __name__ == "__main__":
    main()
