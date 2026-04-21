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


def train(data_yaml, epochs=100, model_name="yolov8n.pt", imgsz=640, batch=16):
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
        patience=25,
        # Augmentations useful for shrimp in water
        augment=True,
        mosaic=1.0,
        fliplr=0.5,
        flipud=0.5,
        degrees=45.0,
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
