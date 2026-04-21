"""Count tiny post-larvae shrimp using optimised classical CV config."""
import sys, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, '.')
from src.preprocessing import resize_frame, preprocess_for_detection
from src.classical_counter import _detect_contours
from tqdm import tqdm

VIDEO   = 'data/videos/tomnho.mp4'
OUT_VID = 'outputs/tomnho_classical_annotated.mp4'
SKIP    = 5          # process every 6th frame (faster on 9000-frame video)

# Best config for tiny post-larvae
CFG = dict(min_area=8, max_area=600, blur=5, block=15, c=3, morph=3)

cap   = cv2.VideoCapture(VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps   = cap.get(cv2.CAP_PROP_FPS)
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
RW = 640
RH = int(orig_h * RW / orig_w)

writer = cv2.VideoWriter(OUT_VID, cv2.VideoWriter_fourcc(*'mp4v'), fps, (RW, RH))

frame_counts = []
idx = 0

with tqdm(total=total, desc="Counting tiny shrimp") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % (SKIP + 1) == 0:
            frame = resize_frame(frame, RW)
            thresh = preprocess_for_detection(frame,
                blur_kernel=CFG['blur'], adaptive_block_size=CFG['block'],
                adaptive_c=CFG['c'], morph_kernel_size=CFG['morph'])
            boxes, _ = _detect_contours(thresh, CFG['min_area'], CFG['max_area'])
            frame_counts.append(len(boxes))

            annotated = frame.copy()
            for x, y, w, h in boxes:
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(annotated, f"Visible: {len(boxes)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            writer.write(annotated)
        idx += 1
        pbar.update(1)

cap.release()
writer.release()

p50  = int(np.percentile(frame_counts, 50))
p90  = int(np.percentile(frame_counts, 90))
pmax = int(np.max(frame_counts))
pavg = round(float(np.mean(frame_counts)), 1)

print()
print("=" * 52)
print("  SMART SHRIMP COUNT — tôm nhỏ (post-larvae)")
print("=" * 52)
print(f"  Frames processed  : {len(frame_counts)}")
print(f"  Avg visible/frame : {pavg}")
print(f"  Median            : {p50}")
print(f"  90th percentile   : {p90}  <- estimated count")
print(f"  Max visible       : {pmax}")
print("=" * 52)
print(f"  Output video: {OUT_VID}")
