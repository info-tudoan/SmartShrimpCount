"""
Extract frames from a video for YOLO training data annotation.

Usage:
  python scripts/extract_frames.py video.mp4
  python scripts/extract_frames.py video.mp4 --output data/frames --every 15 --max 300
"""
import argparse
import cv2
from pathlib import Path
from tqdm import tqdm


def extract_frames(video_path, output_dir="data/frames", every_n=30, max_frames=200, resize_width=640):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(video_path).stem
    saved = 0
    frame_idx = 0
    estimated = min(max_frames, total // max(every_n, 1))

    with tqdm(total=estimated, desc="Extracting") as pbar:
        while saved < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % every_n == 0:
                h, w = frame.shape[:2]
                if resize_width and w > resize_width:
                    frame = cv2.resize(frame, (resize_width, int(h * resize_width / w)))
                cv2.imwrite(str(out_dir / f"{stem}_{frame_idx:06d}.jpg"), frame)
                saved += 1
                pbar.update(1)
            frame_idx += 1

    cap.release()
    print(f"\nExtracted {saved} frames -> {output_dir}")
    print()
    print("Next steps:")
    print("  1. Annotate frames at https://roboflow.com or https://cvat.ai")
    print("     Label every shrimp as class: shrimp")
    print("  2. Export dataset in YOLOv8 format -> data/dataset/")
    print("  3. Train: python scripts/train_yolo.py --data data/dataset/data.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames for YOLO training data")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--output", default="data/frames", help="Output directory")
    parser.add_argument("--every", type=int, default=30, help="Extract every N frames (default: 30)")
    parser.add_argument("--max", type=int, default=200, help="Max frames to extract (default: 200)")
    parser.add_argument("--width", type=int, default=640, help="Resize width (default: 640)")
    args = parser.parse_args()

    extract_frames(args.video, args.output, args.every, args.max, args.width)
