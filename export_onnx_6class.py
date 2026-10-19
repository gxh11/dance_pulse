# export_and_benchmark_onnx.py
import time
import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
from torchvision.models.video import r3d_18

# 1. 加载训练好的 6 分类 PyTorch 权重 (5 舞种 + no_dance)
CLASS_NAMES = ["breakdancing", "ballet_jazz", "locking", "hiphop", "popping", "no_dance"]
model = r3d_18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load("./checkpoints/r3d_dance_6class.pth", map_location="cpu"))
model.eval()

# 2. 导出为 ONNX 格式 (独立文件名, 不覆盖在役的 5 分类 dance_classifier.onnx)
onnx_path = "./checkpoints/dance_classifier_6class.onnx"
dummy_input = torch.randn(1, 3, 16, 112, 112)

export_kwargs = dict(
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['video_input'],
    output_names=['logits'],
    dynamic_axes={'video_input': {0: 'batch_size'}, 'logits': {0: 'batch_size'}},
)
# 新 torch 默认 dynamo 寇040器降到 opset14 会转换失败, 用旧版 TorchScript 寇040器 (dynamo=False)
try:
    torch.onnx.export(model, dummy_input, onnx_path, dynamo=False, **export_kwargs)
except TypeError:
    torch.onnx.export(model, dummy_input, onnx_path, **export_kwargs)  # 旧 torch 无 dynamo 参数
print(f"[*] 成功导出 ONNX 模型至: {onnx_path}")

# 3. 性能基准压测对比 (Benchmark: PyTorch vs ONNX Runtime)
num_runs = 50
np_input = np.random.randn(1, 3, 16, 112, 112).astype(np.float32)

# PyTorch (CPU) 时延
with torch.no_grad():
    for _ in range(5): _ = model(dummy_input)  # 热身
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = model(dummy_input)
    pt_latency = ((time.perf_counter() - start) / num_runs) * 1000

# ONNX Runtime (CPU) 时延
ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
for _ in range(5): _ = ort_session.run(None, {'video_input': np_input})  # 热身
start = time.perf_counter()
for _ in range(num_runs):
    _ = ort_session.run(None, {'video_input': np_input})
ort_latency = ((time.perf_counter() - start) / num_runs) * 1000

print("\n" + "=" * 45)
print("       推理延迟优化压测结果 (CPU 基准)")
print("=" * 45)
print(f"  PyTorch FP32 平均延迟:     {pt_latency:.2f} ms")
print(f"  ONNX Runtime 平均延迟:     {ort_latency:.2f} ms")
print(f"  加速倍率 (Speedup Ratio):  {pt_latency / ort_latency:.2f}x")
print("=" * 45)

# 4. ONNX Runtime CUDA 时延 (服务端实际跑的 EP)
try:
    import torch as _t
    if _t.cuda.is_available():
        sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        for _ in range(5):
            sess.run(None, {'video_input': np_input})
        start = time.perf_counter()
        for _ in range(num_runs):
            sess.run(None, {'video_input': np_input})
        cuda_latency = ((time.perf_counter() - start) / num_runs) * 1000
        print(f"  ONNX Runtime CUDA 平均延迟: {cuda_latency:.2f} ms  @ {sess.get_providers()}")
    else:
        print("  (无 GPU, 跳过 CUDA 基准)")
except Exception as e:
    print(f"  (CUDA 基准失败: {e})")
print("=" * 45)
