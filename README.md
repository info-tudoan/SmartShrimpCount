# SmartShrimp Count — PoC

AI-powered shrimp counting system for De Heus Vietnam.  
Combines **classical Computer Vision** and **YOLOv8 + ByteTrack** to count shrimp directly from video, eliminating manual counting.

---

## Results

| Video | Shrimp Type | Method | Count | Notes |
|-------|-------------|--------|-------|-------|
| WhatsApp Video (De Heus) | Large juvenile | YOLO + ByteTrack | **32 unique** | mAP50 = 0.508 |
| tomnho.mp4 | Tiny post-larvae | Classical CV | **~465 / frame** | 90th pct ≈ 475 |

---

## Project Structure

```
SmartShrimpCount/
├── main.py                    # CLI entry point
├── config.yaml                # Tunable parameters
├── requirements.txt
│
├── src/
│   ├── preprocessing.py       # Resize + adaptive threshold pipeline
│   ├── classical_counter.py   # Contour detection + CentroidTracker
│   ├── yolo_counter.py        # YOLOv8 inference + ByteTrack
│   ├── centroid_tracker.py    # Distance-based unique object counter
│   └── report.py              # JSON report + console summary
│
├── scripts/
│   ├── extract_frames.py      # Extract frames from video for labeling
│   ├── auto_label.py          # SAM-based auto-labeling (large shrimp)
│   ├── auto_label_tiny.py     # CV-based pseudo-labeling (tiny shrimp)
│   ├── train_yolo.py          # YOLOv8 training script
│   ├── tune_config.py         # Grid-search best classical CV config
│   └── count_tomnho.py        # Quick count script for tomnho.mp4
│
├── models/
│   └── best.pt                # Trained YOLOv8n (large shrimp, mAP50=0.508)
│
├── data/
│   ├── videos/                # Input videos
│   ├── frames/                # Extracted frames for labeling
│   └── labeled_tomnho/        # Auto-labeled tiny shrimp dataset
│       ├── images/
│       ├── labels/
│       └── data.yaml
│
└── outputs/                   # Annotated videos + JSON reports
```

---

## Quick Start

### 1. Install dependencies

```bash
# Using uv (recommended on Windows)
uv venv
uv pip install -r requirements.txt
```

Or with pip:

```bash
pip install -r requirements.txt
```

### 2. Count shrimp — Classical CV (no trained model needed)

Works best for **tiny post-larvae shrimp**:

```bash
python main.py --video data/videos/tomnho.mp4
```

### 3. Count shrimp — YOLO + ByteTrack

Works best for **large juvenile shrimp**. Requires `models/best.pt`:

```bash
python main.py --video "data/videos/WhatsApp Video 2026-04-18 at 10.59.17.mp4" --method yolo
```

### 4. Options

```bash
python main.py --video VIDEO_PATH [OPTIONS]

Options:
  --method        classical | yolo  (default: classical)
  --config        Path to config YAML  (default: config.yaml)
  --output        Custom output video path
  --no-output-video  Skip saving annotated video (faster)
  --preview       Show live preview window (press Q to quit)
```

---

## Two-Method Architecture

### Classical CV Pipeline (for tiny shrimp, <10px)

```
Frame → Resize(640px) → Grayscale → GaussianBlur
      → AdaptiveThreshold(BINARY_INV) → MorphOpen+Close
      → FindContours → Filter by area → CentroidTracker
```

**Best config for tiny post-larvae** (`tomnho.mp4`):
```yaml
min_area: 8    max_area: 600
blur: 5        block: 15    c: 3    morph: 3
```

Result: ~465 shrimp visible per frame, 90th percentile ≈ 475.

### YOLO + ByteTrack Pipeline (for large shrimp, >20px)

```
Frame → Resize(640px) → YOLOv8n inference
      → sv.Detections → ByteTrack.update()
      → Collect unique track IDs → estimated count
```

Model trained via SAM auto-labeling on 65 frames from De Heus WhatsApp video.  
Result: 32 unique shrimp tracked, mAP50 = 0.508.

---

## Training Your Own Model

### Step 1 — Extract frames

```bash
python scripts/extract_frames.py --video data/videos/your_video.mp4 --output data/frames --every 30
```

### Step 2 — Auto-label

**Large shrimp** (uses SAM + SimpleBlobDetector for eye detection):
```bash
python scripts/auto_label.py --frames data/frames --output data/labeled
```

**Tiny post-larvae** (uses classical CV detections as pseudo-labels):
```bash
python scripts/auto_label_tiny.py --frames data/frames --output data/labeled_tiny
```

Verify labels visually in `data/labeled_tiny/review/` before training.

### Step 3 — Train

```bash
python scripts/train_yolo.py --data data/labeled_tiny/data.yaml --model yolov8n.pt --epochs 80
```

Best weights are automatically copied to `models/best.pt`.

---

## Tuning Classical CV Parameters

Run the config tuner on your video to find optimal parameters:

```bash
python scripts/tune_config.py
```

Edit the `configs` list in `tune_config.py` to try different combinations. The best config is saved as an annotated sample image in `outputs/tomnho_tuned.jpg`.

---

## Configuration Reference

```yaml
# config.yaml

classical:
  min_contour_area: 2000   # Min contour size in px² (large shrimp)
  max_contour_area: 80000  # Max contour size in px²
  blur_kernel: 11          # Gaussian blur (must be odd)
  adaptive_block_size: 51  # Larger = merges body parts
  adaptive_c: 8
  morph_kernel_size: 9
  max_disappeared: 40      # Frames before lost track removed
  max_distance: 80         # Max px distance to match same shrimp

yolo:
  model_path: "models/best.pt"
  confidence: 0.25
  iou_threshold: 0.45
  device: "cpu"            # "0" for GPU

video:
  skip_frames: 1           # Process every N+1 frames
  resize_width: 640
  output_dir: "outputs"
```

> **Tip:** For tiny post-larvae, use `count_tomnho.py` directly — it has tuned parameters hardcoded and processes every 6th frame for speed.

---

## Output Format

Each run produces:
- **Annotated video** — bounding boxes + count overlay (`outputs/*.mp4`)
- **JSON report** — structured results (`outputs/*_<method>_<timestamp>.json`)

Example JSON:
```json
{
  "method": "yolo",
  "estimated_count": 32,
  "max_visible": 18,
  "avg_visible": 9.4,
  "unique_track_ids": 32,
  "frames_processed": 450,
  "total_frames": 901,
  "video_path": "data/videos/WhatsApp Video 2026-04-18 at 10.59.17.mp4",
  "model_used": "models/best.pt"
}
```

---

## Roadmap

| Phase | Target | Status |
|-------|--------|--------|
| PoC — Classical CV + YOLO | Prove concept on real De Heus videos | ✅ Done |
| Auto-labeling pipeline | No manual annotation needed | ✅ Done |
| YOLO training — large shrimp | mAP50 ≥ 0.5 on WhatsApp video | ✅ Done (0.508) |
| YOLO training — tiny shrimp | Train on 57k auto-labeled boxes | 🔄 In progress |
| Multi-tank support | Process multiple camera feeds | 📋 Planned |
| Edge deployment | Jetson Nano / Raspberry Pi | 📋 Planned |
| Web dashboard | Real-time count + alerts | 📋 Planned |

---

## Requirements

- Python 3.11+
- OpenCV ≥ 4.10
- ultralytics ≥ 8.3 (YOLOv8/YOLOv11)
- supervision ≥ 0.21 (ByteTrack)
- scipy, numpy, tqdm, pyyaml

GPU optional — CPU inference is fast enough for offline video processing.

---

## Contact

**Infodation** — application-administration@infodation.nl  
PoC developed for **De Heus Vietnam** — SmartShrimp Count pilot project, April 2026.
