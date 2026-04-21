"""Quick config tuner — find best classical CV params for a given video."""
import sys, cv2, numpy as np
sys.path.insert(0, '.')
from src.preprocessing import resize_frame, preprocess_for_detection
from src.classical_counter import _detect_contours

video = 'data/videos/tomnho.mp4'
cap = cv2.VideoCapture(video)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

configs = [
    dict(min_area=5,  max_area=300,  blur=3, block=7,  c=2, morph=2),
    dict(min_area=10, max_area=500,  blur=3, block=9,  c=3, morph=2),
    dict(min_area=3,  max_area=200,  blur=5, block=11, c=2, morph=1),
    dict(min_area=5,  max_area=400,  blur=3, block=11, c=2, morph=2),
    dict(min_area=8,  max_area=600,  blur=5, block=15, c=3, morph=3),
]

sample_frames = [total//6, total//4, total//3, total//2, int(total*2//3)]

print(f"Video: {total} frames")
print(f"Testing {len(configs)} configs on {len(sample_frames)} sample frames\n")

best_cfg, best_avg = None, 0
for cfg in configs:
    counts = []
    for fi in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = resize_frame(frame, 640)
        thresh = preprocess_for_detection(frame,
            blur_kernel=cfg['blur'], adaptive_block_size=cfg['block'],
            adaptive_c=cfg['c'], morph_kernel_size=cfg['morph'])
        boxes, _ = _detect_contours(thresh, cfg['min_area'], cfg['max_area'])
        counts.append(len(boxes))
    avg = np.mean(counts)
    print(f"min={cfg['min_area']:3d} max={cfg['max_area']:4d} blur={cfg['blur']} "
          f"block={cfg['block']} c={cfg['c']} morph={cfg['morph']}  "
          f"->  {counts}  avg={avg:.0f}")
    if avg > best_avg:
        best_avg, best_cfg = avg, cfg

cap.release()
print(f"\nBest config: {best_cfg}  (avg {best_avg:.0f} detections/frame)")

# Save annotated frame with best config
cap2 = cv2.VideoCapture(video)
cap2.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
ret, frame = cap2.read()
cap2.release()
frame = resize_frame(frame, 640)
thresh = preprocess_for_detection(frame,
    blur_kernel=best_cfg['blur'], adaptive_block_size=best_cfg['block'],
    adaptive_c=best_cfg['c'], morph_kernel_size=best_cfg['morph'])
boxes, _ = _detect_contours(thresh, best_cfg['min_area'], best_cfg['max_area'])
annotated = frame.copy()
for x, y, w, h in boxes:
    cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 1)
cv2.putText(annotated, f"Detected: {len(boxes)} shrimp", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
cv2.imwrite('outputs/tomnho_tuned.jpg', annotated)
print("Annotated sample -> outputs/tomnho_tuned.jpg")
