"""
RT-DETR baseline comparison for tennis ball detection.

Trains RT-DETR-l on the same dataset as YOLOv8n for 50 epochs,
then prints a side-by-side comparison of mAP50, precision, recall,
and GPU inference speed.

RT-DETR uses a Vision Transformer encoder (hybrid CNN+ViT) rather than
a pure conv backbone, making it a meaningful architectural contrast to YOLO.

Usage:
    python train_rtdetr.py
"""

from pathlib import Path
from ultralytics import RTDETR


def main():
    ROOT      = Path(__file__).parent
    DATA_YAML = ROOT / "training" / "tennis-ball-detection-6" / "data.yaml"

    assert DATA_YAML.exists(), f"data.yaml not found: {DATA_YAML}"

    print("=" * 60)
    print("Training RT-DETR-l on tennis ball dataset (50 epochs)")
    print("Comparison baseline vs YOLOv8n 100-epoch run")
    print("=" * 60)

    model = RTDETR("rtdetr-l.pt")

    results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=8,          # RT-DETR is heavier than YOLOv8n
        device="0",

        project=str(ROOT / "runs"),
        name="rtdetr_tennis_50ep",
        exist_ok=False,

        lr0=0.0001,       # Transformers typically use lower LR
        cos_lr=True,
        patience=20,

        optimizer="AdamW",
        amp=True,
        workers=8,
        seed=42,
        verbose=True,
        plots=True,
        val=True,
    )

    # ---- Summary ----
    yolo_map50   = 0.819
    yolo_prec    = 0.872
    yolo_recall  = 0.810
    yolo_ms      = 2.2

    rtdetr_map50  = results.results_dict.get("metrics/mAP50(B)", 0)
    rtdetr_prec   = results.results_dict.get("metrics/precision(B)", 0)
    rtdetr_recall = results.results_dict.get("metrics/recall(B)", 0)

    print("\n" + "=" * 60)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'YOLOv8n (100ep)':>18} {'RT-DETR-l (50ep)':>18}")
    print("-" * 60)
    print(f"{'mAP50':<20} {yolo_map50:>18.3f} {rtdetr_map50:>18.3f}")
    print(f"{'Precision':<20} {yolo_prec:>18.3f} {rtdetr_prec:>18.3f}")
    print(f"{'Recall':<20} {yolo_recall:>18.3f} {rtdetr_recall:>18.3f}")
    print(f"{'GPU ms/img (val)':<20} {yolo_ms:>18.1f} {'see val speed':>18}")
    print(f"{'Params':<20} {'3.0M':>18} {'32M+':>18}")
    print(f"{'Backbone':<20} {'CSPDarknet (CNN)':>18} {'ResNet+ViT':>18}")
    print("=" * 60)
    print("\nNote: RT-DETR uses a transformer encoder giving global context")
    print("at the cost of higher compute. For small datasets, YOLO often")
    print("wins on speed while ViT-based models may generalise better.")


if __name__ == "__main__":
    main()
