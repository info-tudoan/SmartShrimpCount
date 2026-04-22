"""
Train YOLOv8 on custom shrimp dataset.

Usage:
  python scripts/train_yolo.py --data data/dataset/data.yaml
  python scripts/train_yolo.py --data data/dataset/data.yaml --epochs 150 --model yolov8s.pt

Expected data.yaml structure:
  path: ../data/dataset
  train: images/train
  val: images/val
  nc: 1
  names: ['shrimp']
"""
import argparse
from pathlib import Path


def train(data_yaml, epochs=100, model_name="yolov8n.pt", imgsz=640, batch=8):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Run: pip install ultralytics")

    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    model = YOLO(model_name)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project="runs/train",
        name="shrimp",
        save=True,
        patience=30,
        # CPU-optimised augmentations: fast transforms only.
        # mosaic=0 because combining 4 images is very slow on CPU (~4x cost).
        # The dataset already has diverse shrimp positions so mosaic is less critical.
        mosaic=0.0,      # disabled — too slow on CPU
        fliplr=0.5,      # horizontal flip (near-free)
        flipud=0.5,      # vertical flip  (near-free)
        hsv_h=0.015,     # hue jitter
        hsv_s=0.7,       # saturation jitter — handles different water colours
        hsv_v=0.4,       # brightness jitter
        scale=0.5,       # random scale ±50%
        translate=0.1,   # random translate ±10%
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    dest = Path("models/best.pt")
    dest.parent.mkdir(exist_ok=True)

    import shutil
    shutil.copy(best_weights, dest)
    print(f"\nTraining complete.")
    print(f"Best weights copied to: {dest}")
    print(f"\nRun counting with trained model:")
    print(f"  python main.py --video your_video.mp4 --method yolo")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 for shrimp detection")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--model", default="yolov8n.pt", help="Base model (yolov8n/s/m.pt)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    train(args.data, args.epochs, args.model, args.imgsz, args.batch)
