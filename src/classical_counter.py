import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from .preprocessing import resize_frame, preprocess_for_detection
from .centroid_tracker import CentroidTracker


def _detect_contours(thresh, min_area, max_area):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes, centroids = [], []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))
            centroids.append((int(x + w / 2), int(y + h / 2)))
    return boxes, centroids


def count_shrimp_classical(video_path, config, output_path=None, show_preview=False):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = max(cap.get(cv2.CAP_PROP_FPS), 1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cfg = config["classical"]
    vid_cfg = config["video"]
    resize_w = vid_cfg.get("resize_width", 640)
    skip = vid_cfg.get("skip_frames", 1)

    tracker = CentroidTracker(
        max_disappeared=cfg["max_disappeared"],
        max_distance=cfg["max_distance"],
    )

    writer = None
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out_h = int(orig_h * resize_w / orig_w) if resize_w else orig_h
        out_w = resize_w or orig_w
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    frame_visible_counts = []
    frame_idx = 0

    with tqdm(total=total_frames, desc="[Classical] Processing") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % max(skip, 1) != 0:
                frame_idx += 1
                pbar.update(1)
                continue

            frame = resize_frame(frame, resize_w)
            thresh = preprocess_for_detection(
                frame,
                blur_kernel=cfg["blur_kernel"],
                adaptive_block_size=cfg["adaptive_block_size"],
                adaptive_c=cfg["adaptive_c"],
                morph_kernel_size=cfg["morph_kernel_size"],
            )

            boxes, centroids = _detect_contours(thresh, cfg["min_contour_area"], cfg["max_contour_area"])
            active, total_tracked = tracker.update(centroids)
            frame_visible_counts.append(len(active))

            if writer or show_preview:
                annotated = frame.copy()
                for x, y, w, h in boxes:
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 1)
                for obj_id, (cx, cy) in active.items():
                    cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)
                    cv2.putText(annotated, str(obj_id), (cx + 4, cy - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 255), 1)
                label = f"Visible: {len(active)}  |  Total tracked: {total_tracked}"
                cv2.putText(annotated, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                if writer:
                    writer.write(annotated)
                if show_preview:
                    cv2.imshow("SmartShrimp - Classical", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            frame_idx += 1
            pbar.update(1)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Best estimate: 90th percentile of visible count per frame.
    # Higher than average (accounts for frames where shrimp are obscured),
    # more stable than max (outlier-resistant).
    estimated = int(np.percentile(frame_visible_counts, 90)) if frame_visible_counts else 0

    return {
        "method": "classical",
        "estimated_count": estimated,
        "max_visible": int(max(frame_visible_counts)) if frame_visible_counts else 0,
        "avg_visible": round(float(np.mean(frame_visible_counts)), 1) if frame_visible_counts else 0,
        "total_tracked_ids": tracker.total_registered,
        "frames_processed": len(frame_visible_counts),
        "total_frames": total_frames,
        "video_path": str(video_path),
        "output_video": str(output_path) if output_path else None,
    }
