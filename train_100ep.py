"""
100-epoch fine-tune of YOLOv8n for tennis ball detection.

Resumes from the 5-epoch checkpoint (last.pt) with improved settings:
  - Cosine LR schedule for smoother convergence over 100 epochs
  - Lower initial LR since we are continuing from a partial checkpoint
  - Stronger scale augmentation to help with small/fast-moving balls
  - Slight rotation and mixup regularization for the small dataset (~430 images)
  - Early stopping patience of 30 epochs
  - Named separately so the 5-epoch run results are preserved
"""

from pathlib import Path
from ultralytics import YOLO


def main():
    ROOT = Path(__file__).parent
    DATA_YAML = ROOT / "training" / "tennis-ball-detection-6" / "data.yaml"
    LAST_PT   = ROOT / "runs" / "tennis_ball_train" / "weights" / "last.pt"

    assert DATA_YAML.exists(), f"data.yaml not found: {DATA_YAML}"
    assert LAST_PT.exists(),   f"last.pt not found: {LAST_PT}"

    model = YOLO(str(LAST_PT))

    results = model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=16,
        device="0",

        project=str(ROOT / "runs"),
        name="tennis_ball_100ep",
        exist_ok=False,

        # LR: start lower since we are continuing from a partial checkpoint
        lr0=0.002,
        lrf=0.01,
        cos_lr=True,

        # Early stopping: stop if val mAP50 doesn't improve for 30 epochs
        patience=30,

        # Augmentation tuned for small, fast-moving objects on varied backgrounds
        scale=0.7,        # larger scale jitter for scale robustness
        degrees=10.0,     # slight rotation (ball has no orientation constraint)
        mixup=0.1,        # light mixup regularisation for small dataset
        close_mosaic=20,  # keep mosaic active longer before switching off
        flipud=0.0,       # vertical flip unhelpful for real video footage

        optimizer="auto",
        amp=True,
        workers=8,
        seed=42,
        verbose=True,
        plots=True,
        val=True,
    )

    print("\n=== Training complete ===")
    best = ROOT / "runs" / "tennis_ball_100ep" / "weights" / "best.pt"
    print(f"Best weights : {best}")
    print(f"mAP50        : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"Precision    : {results.results_dict.get('metrics/precision(B)', 'N/A'):.4f}")
    print(f"Recall       : {results.results_dict.get('metrics/recall(B)', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
