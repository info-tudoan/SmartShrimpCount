"""
Auto-label tiny post-larvae shrimp using classical CV detections as YOLO labels.
For small shrimp (<10px), classical CV + contour detection is more reliable than SAM.

Usage:
  python scripts/auto_label_tiny.py
  python scripts/auto_label_tiny.py --frames data/frames_tomnho --output data/labeled_tomnho
"""
import argparse, shutil, sys, cv2, numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '.')
from src.preprocessing import resize_frame, preprocess_for_detection
from src.classical_counter import _detect_contours

# Best config for tiny post-larvae (tuned earlier)
DEFAULT_CFG = dict(min_area=8, max_area=600, blur=5, block=15, c=3, morph=3)


def boxes_to_yolo(boxes_xywh, img_w, img_h):
    """Convert list of (x,y,w,h) pixel boxes to YOLO normalised format."""
    lines = []
    for x, y, w, h in boxes_xywh:
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def nms_boxes(boxes_xywh, iou_thresh=0.3):
    """Remove overlapping boxes (same shrimp detected twice)."""
    if not boxes_xywh:
        return []
    bx = np.array([[x, y, x + w, y + h] for x, y, w, h in boxes_xywh], dtype=float)
    areas = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
    order = areas.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(bx[i, 0], bx[order[1:], 0])
        yy1 = np.maximum(bx[i, 1], bx[order[1:], 1])
        xx2 = np.minimum(bx[i, 2], bx[order[1:], 2])
        yy2 = np.minimum(bx[i, 3], bx[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_thresh]
    return [boxes_xywh[i] for i in keep]


def auto_label_tiny(frames_dir, output_dir, cfg=None):
    if cfg is None:
        cfg = DEFAULT_CFG

    frames_dir = Path(frames_dir)
    labels_dir = Path(output_dir) / "labels"
    images_dir = Path(output_dir) / "images"
    review_dir = Path(output_dir) / "review"
    for d in (labels_dir, images_dir, review_dir):
        d.mkdir(parents=True, exist_ok=True)

    images = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if not images:
        print(f"No images in {frames_dir}")
        return

    total_boxes = 0
    per_frame   = []

    for img_path in tqdm(images, desc="Auto-labeling (tiny shrimp)"):
        frame = cv2.imread(str(img_path))
        img_h, img_w = frame.shape[:2]

        thresh = preprocess_for_detection(frame,
            blur_kernel=cfg['blur'], adaptive_block_size=cfg['block'],
            adaptive_c=cfg['c'], morph_kernel_size=cfg['morph'])

        boxes, _ = _detect_contours(thresh, cfg['min_area'], cfg['max_area'])
        boxes    = nms_boxes(boxes)

        yolo_lines = boxes_to_yolo(boxes, img_w, img_h)

        # Save label
        (labels_dir / (img_path.stem + ".txt")).write_text("\n".join(yolo_lines))
        shutil.copy(img_path, images_dir / img_path.name)

        # Review image
        vis = frame.copy()
        for x, y, w, h in boxes:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(vis, f"Labeled: {len(boxes)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imwrite(str(review_dir / img_path.name), vis)

        total_boxes += len(boxes)
        per_frame.append(len(boxes))

    print(f"\nDone: {len(images)} frames")
    print(f"  Total annotations  : {total_boxes}")
    print(f"  Avg per frame      : {np.mean(per_frame):.0f}")
    print(f"  Min / Max          : {min(per_frame)} / {max(per_frame)}")
    print(f"  Labels             : {labels_dir}")
    print(f"  Review             : {review_dir}  <- verify here")

    # Write data.yaml
    yaml = (f"path: {Path(output_dir).resolve()}\n"
            f"train: images\nval: images\nnc: 1\nnames: ['shrimp']\n")
    yaml_path = Path(output_dir) / "data.yaml"
    yaml_path.write_text(yaml)
    print(f"  data.yaml          : {yaml_path}")
    print()
    print("Next: python scripts/train_yolo.py --data "
          f"{output_dir}/data.yaml --model yolov8n.pt --epochs 80")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="data/frames_tomnho")
    parser.add_argument("--output", default="data/labeled_tomnho")
    args = parser.parse_args()
    auto_label_tiny(args.frames, args.output)
