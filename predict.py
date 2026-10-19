# predict.py — 用已训练好的模型对一个新视频预测舞种
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

CLASS_NAMES = ["breakdancing", "ballet_jazz", "locking", "hiphop", "popping"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_FRAMES = 16   # 与训练一致: 16 帧 (~2 秒)
INPUT_SIZE = 112   # 与训练一致
WEIGHTS = "checkpoints/r3d_dance_best.pth"   # 训练好的权重

def extract_clip(vpath, num_frames=CLIP_FRAMES):
    """与训练完全一致的 seek 采样: 均匀取 16 帧, 缩放到 112x112"""
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

def normalize_clip(clip):
    """与训练完全一致的归一化: -> (C, T, H, W) float32"""
    arr = clip.astype(np.float32) / 255.0
    arr = (arr - np.float32([0.45, 0.45, 0.45])) / np.float32([0.225, 0.225, 0.225])
    return torch.from_numpy(arr.astype(np.float32)).permute(3, 0, 1, 2)

def predict(video_path):
    # 1. 构建模型结构并加载已训练权重
    model = r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    model = model.to(DEVICE).eval()

    # 2. 视频 -> 16 帧片段 -> 归一化
    clip = extract_clip(video_path)
    x = normalize_clip(clip).unsqueeze(0).to(DEVICE)  # (1, C, T, H, W)

    # 3. 推理
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    pred = int(probs.argmax())
    return pred, probs

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("用法: python predict.py <视频路径>")
    video = sys.argv[1]
    if not os.path.exists(video):
        raise SystemExit(f"[!] 找不到视频: {video}")

    pred, probs = predict(video)
    print(f"\n视频: {video}")
    print(f"预测舞种: {CLASS_NAMES[pred]}  (置信度 {probs[pred]*100:.2f}%)")
    print("各类别概率:")
    for name, p in zip(CLASS_NAMES, probs):
        bar = "#" * int(round(p * 30))
        print(f"  {name:<12} {p*100:6.2f}%  {bar}")
