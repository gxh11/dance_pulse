# inference_engine.py
import numpy as np
import onnxruntime as ort

CLASSES = ["breakdancing", "ballet_jazz", "locking", "hiphop", "popping", "no_dance"]


def _cuda_available() -> bool:
    """探测 CUDA 设备。无 GPU 时若强行创建 CUDA EP 会话会直接段错误
    (进程静默崩溃, try/except 拦不住), 因此必须先探测再选 provider。"""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


class DanceInferenceEngine:
    def __init__(self, onnx_model_path="./checkpoints/dance_classifier_6class.onnx", threshold=0.45):
        # 有 GPU 用 CUDA, 否则 CPU (容器无 --gpus 时同样走 CPU)
        if _cuda_available():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.threshold = threshold

    def predict(self, frames_np: np.ndarray) -> dict:
        """
        frames_np: 形状为 (16, 112, 112, 3)，取值 0~255 的连续帧数组
        """
        # 1. 预处理与标准化
        tensor = frames_np.astype(np.float32) / 255.0
        mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
        std = np.array([0.225, 0.225, 0.225], dtype=np.float32)
        tensor = (tensor - mean) / std

        # (T, H, W, C) -> (1, C, T, H, W)
        tensor = np.transpose(tensor, (3, 0, 1, 2))[np.newaxis, ...]

        # 2. ONNX Runtime 推理
        logits = self.session.run(None, {self.input_name: tensor})[0][0]

        # 3. 计算 Softmax 概率
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        max_idx = int(np.argmax(probs))
        max_prob = float(probs[max_idx])

        # 4. Unknown / No Dance Fallback 保护
        if max_prob < self.threshold:
            pred_genre = "Unknown / No Dance"
        else:
            pred_genre = CLASSES[max_idx]

        return {
            "genre": pred_genre,
            "confidence": round(max_prob, 4),
            "distribution": {CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CLASSES))}
        }