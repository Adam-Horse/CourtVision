"""
Benchmark PyTorch vs ONNX Runtime inference speed for the fine-tuned tennis ball detector.

Exports best.pt to ONNX if not already done, then times both backends
over N warmup + M timed runs on a real validation image.

Usage:
    python benchmark_onnx.py
"""

import time
from pathlib import Path

import numpy as np
import cv2


BEST_PT   = Path(__file__).parent / "runs" / "tennis_ball_100ep-4" / "weights" / "best.pt"
ONNX_PATH = BEST_PT.with_suffix(".onnx")
VAL_DIR   = Path(__file__).parent / "training" / "tennis-ball-detection-6" / "tennis-ball-detection-6" / "valid" / "images"
WARMUP    = 20
RUNS      = 200
IMGSZ     = 640


def pick_image():
    images = list(VAL_DIR.glob("*.jpg")) + list(VAL_DIR.glob("*.png"))
    assert images, f"No images found in {VAL_DIR}"
    return images[0]


def preprocess(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    img = cv2.resize(img, (IMGSZ, IMGSZ))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img.transpose(2, 0, 1)[None]  # (1, 3, H, W)


def benchmark_pytorch(img_np: np.ndarray):
    import torch
    from ultralytics import YOLO

    model = YOLO(str(BEST_PT))
    net = model.model.cuda().eval()
    img_tensor = torch.from_numpy(img_np).cuda()

    # Warmup
    for _ in range(WARMUP):
        with torch.no_grad():
            net(img_tensor)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(RUNS):
        with torch.no_grad():
            net(img_tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return (elapsed / RUNS) * 1000  # ms per image


def export_onnx():
    from ultralytics import YOLO
    if ONNX_PATH.exists():
        print(f"ONNX model already exists: {ONNX_PATH}")
        return
    print("Exporting to ONNX...")
    model = YOLO(str(BEST_PT))
    model.export(format="onnx", imgsz=IMGSZ, simplify=True, opset=12)
    # Ultralytics saves next to the .pt file
    exported = BEST_PT.with_suffix(".onnx")
    assert exported.exists(), f"Export failed, expected {exported}"
    print(f"Exported: {exported}")


def benchmark_onnx(img_np: np.ndarray):
    import onnxruntime as ort

    providers = ort.get_available_providers()
    provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in providers else "CPUExecutionProvider"
    print(f"ONNX provider: {provider}")

    sess = ort.InferenceSession(str(ONNX_PATH), providers=[provider])
    input_name = sess.get_inputs()[0].name

    # Warmup
    for _ in range(WARMUP):
        sess.run(None, {input_name: img_np})

    t0 = time.perf_counter()
    for _ in range(RUNS):
        sess.run(None, {input_name: img_np})
    elapsed = time.perf_counter() - t0
    return (elapsed / RUNS) * 1000  # ms per image


def main():
    assert BEST_PT.exists(), f"Weights not found: {BEST_PT}"

    img_path = pick_image()
    print(f"Benchmark image: {img_path.name}")
    img_np = preprocess(img_path)

    print(f"\nRunning {WARMUP} warmup + {RUNS} timed iterations per backend\n")

    print("--- PyTorch (CUDA) ---")
    pt_ms = benchmark_pytorch(img_np)
    print(f"  {pt_ms:.2f} ms/image")

    export_onnx()

    print("\n--- ONNX Runtime ---")
    onnx_ms = benchmark_onnx(img_np)
    print(f"  {onnx_ms:.2f} ms/image")

    speedup = pt_ms / onnx_ms if onnx_ms > 0 else float("inf")
    print(f"\n=== Results ===")
    print(f"PyTorch CUDA : {pt_ms:.2f} ms/image")
    print(f"ONNX Runtime : {onnx_ms:.2f} ms/image")
    print(f"Speedup      : {speedup:.2f}x ({'ONNX faster' if speedup > 1 else 'PyTorch faster'})")
    print(f"\nEquivalent throughput:")
    print(f"  PyTorch  : {1000/pt_ms:.0f} images/sec")
    print(f"  ONNX     : {1000/onnx_ms:.0f} images/sec")


if __name__ == "__main__":
    main()
