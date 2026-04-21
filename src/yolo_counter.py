import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from ultralytics import YOLO
    import supervision as sv
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False


def count_shrimp_yolo(video_path, config, output_path=None, show_preview=False):
    if not _DEPS_OK:
        raise ImportError("Run: pip install ultralytics supervision")

    cfg = config["yolo"]
    vid_cfg = config["video"]
    resize_w = vid_cfg.get("resize_width", 640)
    skip = vid_cfg.get("skip_frames", 1)

    model_path = cfg.get("model_path", "models/best.pt")
    if not Path(model_path).exists():
        # Generic nano model for pipeline validation — won't detect shrimp
        # until a custom model is trained (see scripts/train_yolo.py)
        print(f"[WARN] Model not found: {model_path}")
        print("[WARN] Falling back to yolov8n.pt (NOT trained for shrimp — counts will be meaningless)")
        model_path = "yolov8n.pt"

    model = YOLO(model_path)
    tracker = sv.ByteTrack()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = max(cap.get(cv2.CAP_PROP_FPS), 1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out_h = int(orig_h * resize_w / orig_w) if resize_w else orig_h
        out_w = resize_w or orig_w
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    box_annotator = sv.BoxAnnotator(thickness=1)
    label_annotator = sv.LabelAnnotator(text_scale=0.3)

    unique_ids = set()
    frame_counts = []
    frame_idx = 0

    with tqdm(total=total_frames, desc="[YOLO] Processing") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % max(skip, 1) != 0:
                frame_idx += 1
                pbar.update(1)
                continue

            h, w = frame.shape[:2]
            if resize_w and w != resize_w:
                frame = cv2.resize(frame, (resize_w, int(h * resize_w / w)))

            results = model(
                frame,
                conf=cfg["confidence"],
                iou=cfg["iou_threshold"],
                device=cfg.get("device", "cpu"),
                verbose=False,
            )

            detections = sv.Detections.from_ultralytics(results[0])
            tracks = tracker.update_with_detections(detections)

            if tracks.tracker_id is not None:
                for tid in tracks.tracker_id:
                    unique_ids.add(int(tid))

            frame_counts.append(len(tracks))

            if writer or show_preview:
                tids = tracks.tracker_id if tracks.tracker_id is not None else []
                labels = [f"#{tid}" for tid in tids]
                annotated = box_annotator.annotate(frame.copy(), tracks)
                annotated = label_annotator.annotate(annotated, tracks, labels)
                label = f"Visible: {len(tracks)}  |  Unique IDs: {len(unique_ids)}"
                cv2.putText(annotated, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                if writer:
                    writer.write(annotated)
                if show_preview:
                    cv2.imshow("SmartShrimp - YOLO", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            frame_idx += 1
            pbar.update(1)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    return {
        "method": "yolo",
        "estimated_count": len(unique_ids),
        "max_visible": int(max(frame_counts)) if frame_counts else 0,
        "avg_visible": round(float(np.mean(frame_counts)), 1) if frame_counts else 0,
        "unique_track_ids": len(unique_ids),
        "frames_processed": len(frame_counts),
        "total_frames": total_frames,
        "video_path": str(video_path),
        "model_used": str(model_path),
        "output_video": str(output_path) if output_path else None,
    }
