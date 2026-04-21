"""
Auto-label shrimp frames using SAM (Segment Anything Model) — runs 100% locally.

Pipeline:
  1. Blob detector  →  finds shrimp eye/head positions (dark round dots)
  2. SAM            →  expands each eye point into a full shrimp mask
  3. Mask → bbox    →  convert to YOLO-format bounding boxes
  4. Review images  →  saved so you can visually verify quality

Usage:
  python scripts/auto_label.py
  python scripts/auto_label.py --frames data/frames --output data/labeled --model mobile_sam.pt
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ── Blob detector: finds shrimp eyes (small dark circles) ─────────────────────

def detect_eye_points(gray):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 30
    params.maxArea = 700
    params.filterByCircularity = True
    params.minCircularity = 0.35
    params.filterByConvexity = True
    params.minConvexity = 0.65
    params.blobColor = 0          # dark blobs
    params.minDistBetweenBlobs = 18
    detector = cv2.SimpleBlobDetector_create(params)
    kps = detector.detect(gray)
    return [(int(k.pt[0]), int(k.pt[1])) for k in kps]


# ── Build rough bounding boxes around each eye point ──────────────────────────

def eye_to_box(cx, cy, img_w, img_h, pad_x=55, pad_y=90):
    """Expand an eye point into a rough shrimp bounding box."""
    x1 = max(0, cx - pad_x)
    y1 = max(0, cy - pad_y)
    x2 = min(img_w, cx + pad_x)
    y2 = min(img_h, cy + pad_y)
    return [x1, y1, x2, y2]


# ── Non-maximum suppression to deduplicate overlapping boxes ──────────────────

def nms_boxes(boxes_xyxy, iou_thresh=0.3):
    if not boxes_xyxy:
        return []
    boxes = np.array(boxes_xyxy, dtype=float)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_thresh]
    return [boxes_xyxy[i] for i in keep]


def merge_nearby_boxes(yolo_boxes, img_w, img_h, merge_dist_px=70):
    """Merge boxes whose centers are closer than merge_dist_px (same shrimp, 2 eyes)."""
    if not yolo_boxes:
        return []
    centers = np.array([(bx * img_w, by * img_h) for bx, by, bw, bh in yolo_boxes])
    used = [False] * len(yolo_boxes)
    merged = []
    for i in range(len(yolo_boxes)):
        if used[i]:
            continue
        cluster = [i]
        for j in range(i + 1, len(yolo_boxes)):
            if not used[j]:
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < merge_dist_px:
                    cluster.append(j)
                    used[j] = True
        used[i] = True
        # Keep the box with the largest area
        best = max(cluster, key=lambda idx: yolo_boxes[idx][2] * yolo_boxes[idx][3])
        merged.append(yolo_boxes[best])
    return merged


# ── Convert mask to YOLO bbox (normalized) ────────────────────────────────────

def mask_to_yolo(mask, img_h, img_w):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cx = (cmin + cmax) / 2 / img_w
    cy = (rmin + rmax) / 2 / img_h
    w  = (cmax - cmin) / img_w
    h  = (rmax - rmin) / img_h
    return cx, cy, w, h


# ── Main labeling loop ─────────────────────────────────────────────────────────

def auto_label(frames_dir, output_dir, model_name="mobile_sam.pt", min_area=0.002, max_area=0.18):
    try:
        from ultralytics import SAM
    except ImportError:
        raise ImportError("Run: pip install ultralytics")

    print(f"Loading SAM model: {model_name}  (downloads ~40 MB on first run)")
    sam = SAM(model_name)

    frames_dir = Path(frames_dir)
    labels_dir = Path(output_dir) / "labels"
    images_dir = Path(output_dir) / "images"
    review_dir = Path(output_dir) / "review"
    for d in (labels_dir, images_dir, review_dir):
        d.mkdir(parents=True, exist_ok=True)

    images = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if not images:
        print(f"No images found in {frames_dir}")
        return

    total_boxes = 0

    for img_path in tqdm(images, desc="Auto-labeling"):
        frame = cv2.imread(str(img_path))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_h, img_w = frame.shape[:2]

        # Step 1: find eye candidates
        eye_pts = detect_eye_points(gray)
        if not eye_pts:
            # Save empty label file so YOLO knows this image has no shrimp
            open(labels_dir / (img_path.stem + ".txt"), "w").close()
            shutil.copy(img_path, images_dir / img_path.name)
            continue

        # Step 2: build rough boxes, deduplicate
        raw_boxes = [eye_to_box(cx, cy, img_w, img_h) for cx, cy in eye_pts]
        boxes = nms_boxes(raw_boxes, iou_thresh=0.5)

        # Step 3: SAM refines each rough box → precise mask
        yolo_lines = []
        vis = frame.copy()

        results = sam(frame, bboxes=boxes, verbose=False)

        refined_boxes = []
        if results and results[0].masks is not None:
            raw_yolo = []
            for mask_tensor in results[0].masks.data:
                mask = mask_tensor.cpu().numpy().astype(bool)
                bbox = mask_to_yolo(mask, img_h, img_w)
                if bbox is None:
                    continue
                bx, by, bw, bh = bbox
                area = bw * bh
                if not (min_area < area < max_area):
                    continue
                aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
                if aspect > 12:
                    continue
                raw_yolo.append(bbox)

            # Merge boxes for the same shrimp (2 eyes → 1 box)
            refined_boxes = merge_nearby_boxes(raw_yolo, img_w, img_h, merge_dist_px=70)

            for bx, by, bw, bh in refined_boxes:
                yolo_lines.append(f"0 {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}")
                x1 = int((bx - bw/2) * img_w)
                y1 = int((by - bh/2) * img_h)
                x2 = int((bx + bw/2) * img_w)
                y2 = int((by + bh/2) * img_h)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            # SAM returned no masks → fall back to rough boxes
            for box in boxes:
                x1,y1,x2,y2 = [int(v) for v in box]
                bx = (x1+x2)/2/img_w; by = (y1+y2)/2/img_h
                bw = (x2-x1)/img_w;   bh = (y2-y1)/img_h
                yolo_lines.append(f"0 {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}")
                cv2.rectangle(vis, (x1,y1), (x2,y2), (0,165,255), 2)

        # Save YOLO label
        label_path = labels_dir / (img_path.stem + ".txt")
        label_path.write_text("\n".join(yolo_lines))

        # Copy image to dataset
        shutil.copy(img_path, images_dir / img_path.name)

        # Save review image
        count = len(yolo_lines)
        total_boxes += count
        cv2.putText(vis, f"Auto-labeled: {count} shrimp", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imwrite(str(review_dir / img_path.name), vis)

    print(f"\nDone: {len(images)} frames, {total_boxes} total annotations")
    print(f"  Labels  : {labels_dir}")
    print(f"  Images  : {images_dir}")
    print(f"  Review  : {review_dir}  <- check these images!")
    print()
    _write_data_yaml(output_dir)
    print("Next step:")
    print(f"  python scripts/train_yolo.py --data {output_dir}/data.yaml")


def _write_data_yaml(output_dir):
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    yaml_content = f"""path: {output_dir.resolve()}
train: images
val: images

nc: 1
names: ['shrimp']
"""
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"  data.yaml: {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-label shrimp frames using SAM (local, no cloud)")
    parser.add_argument("--frames", default="data/frames",     help="Folder with extracted frames")
    parser.add_argument("--output", default="data/labeled",    help="Output folder for labels + review")
    parser.add_argument("--model",  default="mobile_sam.pt",   help="SAM model (mobile_sam.pt or sam_b.pt)")
    args = parser.parse_args()

    auto_label(args.frames, args.output, args.model)
