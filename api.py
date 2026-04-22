"""
SmartShrimp Count — REST API
POST /count        -> Upload video, get shrimp count
POST /count-image  -> Upload image, get shrimp count (fast, single frame)

Usage:
  uvicorn api:app --host 0.0.0.0 --port 8000

Examples:
  curl -X POST http://localhost:8000/count \
       -F "video=@whatsapp.mp4" -F "shrimp_type=large"

  curl -X POST http://localhost:8000/count-image \
       -F "image=@frame.jpg" -F "shrimp_type=large"
"""
import io
import shutil
import tempfile
import time
import yaml
import numpy as np
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(
    title="SmartShrimp Count API",
    description="""
## AI-powered shrimp counting for De Heus Vietnam

### Endpoints:
| Endpoint | Input | Thời gian |
|----------|-------|-----------|
| `POST /count` | Video (mp4/avi/mov) | ~1–3 phút |
| `POST /count-image` | Hình ảnh (jpg/png) | **~2 giây** |

### Chọn đúng `shrimp_type`:
| shrimp_type | Loại tôm | Phương pháp |
|-------------|----------|-------------|
| `large` | Tôm giống lớn (>20px) | YOLO detection |
| `tiny` | Tôm post-larvae nhỏ (<10px) | Classical CV |
""",
    version="1.2.0",
)

# Config tuned cho tiny post-larvae shrimp
TINY_CFG = dict(min_area=8, max_area=600, blur=5, block=15, c=3, morph=3)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_tiny_config(base_config: dict) -> dict:
    """Override classical params with tiny-shrimp tuned values."""
    cfg = base_config.copy()
    cfg["classical"] = {
        "min_contour_area": TINY_CFG["min_area"],
        "max_contour_area": TINY_CFG["max_area"],
        "blur_kernel": TINY_CFG["blur"],
        "adaptive_block_size": TINY_CFG["block"],
        "adaptive_c": TINY_CFG["c"],
        "morph_kernel_size": TINY_CFG["morph"],
        "max_disappeared": 20,
        "max_distance": 40,
    }
    return cfg


@app.get("/")
def root():
    return {
        "service": "SmartShrimp Count API",
        "version": "1.1.0",
        "usage": {
            "large_shrimp": "POST /count with shrimp_type=large  (YOLO, ~3-5 min)",
            "tiny_shrimp":  "POST /count with shrimp_type=tiny   (Classical CV, ~2 min)",
        },
        "docs": "GET /docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/count")
async def count_shrimp(
    video: UploadFile = File(..., description="Video file (mp4, avi, mov, mkv)"),
    shrimp_type: str = Form(
        default="large",
        description=(
            "'large' — tôm giống lớn, dùng YOLO+ByteTrack (chính xác ~3-5 phút) | "
            "'tiny'  — tôm post-larvae nhỏ, dùng Classical CV (~2 phút)"
        ),
    ),
):
    """
    Đếm số tôm trong video.

    ### Chọn shrimp_type phù hợp:
    - **large** → YOLO + ByteTrack, dành cho tôm giống lớn (video De Heus thực tế)
    - **tiny** → Classical CV, dành cho tôm post-larvae siêu nhỏ (<10px/frame)

    ### Lưu ý thời gian xử lý:
    - `large` (YOLO): ~3–5 phút tùy độ dài video
    - `tiny` (Classical): ~1–2 phút
    """

    if shrimp_type not in ("large", "tiny"):
        raise HTTPException(
            status_code=400,
            detail="shrimp_type phải là 'large' (tôm lớn, dùng YOLO) hoặc 'tiny' (tôm nhỏ, dùng Classical CV)",
        )

    allowed_ext = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
    suffix = Path(video.filename or "video.mp4").suffix.lower()
    if suffix not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng '{suffix}' không hỗ trợ. Dùng: {', '.join(allowed_ext)}",
        )

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_video = tmp_dir / f"upload{suffix}"

    try:
        with open(tmp_video, "wb") as f:
            shutil.copyfileobj(video.file, f)

        base_config = load_config()
        start_time = time.time()

        if shrimp_type == "large":
            # YOLO + ByteTrack — chính xác cho tôm lớn
            from src.yolo_counter import count_shrimp_yolo
            result = count_shrimp_yolo(
                tmp_video,
                base_config,
                output_path=None,
                show_preview=False,
            )
            method_used = "yolo+bytetrack"

        else:
            # Classical CV tuned cho tiny post-larvae
            from src.classical_counter import count_shrimp_classical
            tiny_config = build_tiny_config(base_config)
            result = count_shrimp_classical(
                tmp_video,
                tiny_config,
                output_path=None,
                show_preview=False,
            )
            method_used = "classical-cv (tiny-tuned)"

        elapsed = round(time.time() - start_time, 1)

        count_result = result["estimated_count"]
        max_visible = result.get("max_visible", 0)
        avg_visible = result.get("avg_visible", 0)

        # Confidence score: avg_visible / max_visible
        # Measures detection stability across the whole video.
        # High score = model consistently sees nearly all shrimp every frame → reliable count.
        # Low score  = count spikes in a few frames only → result less trustworthy.
        # e.g. avg=21.7, max=27 → 0.80 = "80% of frames show near-maximum shrimp count"
        confidence_score = round(
            avg_visible / max(max_visible, 1), 2
        ) if max_visible > 0 else 0.0

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "filename": video.filename,
                "shrimp_type": shrimp_type,
                "method": method_used,
                "count_result": count_result,
                "confidence_score": confidence_score,
                "avg_visible_per_frame": result.get("avg_visible", 0),
                "max_visible_per_frame": max_visible,
                "frames_processed": result.get("frames_processed", 0),
                "processing_time_seconds": elapsed,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/count-image")
async def count_shrimp_image(
    image: UploadFile = File(..., description="Image file (jpg, jpeg, png, bmp)"),
    shrimp_type: str = Form(
        default="large",
        description="'large' — YOLO detection | 'tiny' — Classical CV",
    ),
    annotated: bool = Form(
        default=False,
        description="If true, returns annotated image instead of JSON",
    ),
):
    """
    Dem so tom trong 1 anh. Nhanh hon video (~2 giay).

    - **large** -> YOLO detect tren 1 frame, tra ve so box phat hien duoc
    - **tiny**  -> Classical CV (adaptive threshold + contour) tren 1 frame
    - **annotated=true** -> tra ve anh JPG co ve box/contour thay vi JSON
    """
    if shrimp_type not in ("large", "tiny"):
        raise HTTPException(
            status_code=400,
            detail="shrimp_type phai la 'large' hoac 'tiny'",
        )

    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    suffix = Path(image.filename or "image.jpg").suffix.lower()
    if suffix not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Dinh dang '{suffix}' khong ho tro. Dung: {', '.join(allowed_ext)}",
        )

    try:
        import cv2
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Khong doc duoc anh — file bi loi hoac khong phai anh hop le")

        base_config = load_config()
        start_time = time.time()

        if shrimp_type == "large":
            # ── YOLO single-frame detection ──────────────────────────────
            from ultralytics import YOLO
            cfg = base_config["yolo"]

            model_path = cfg.get("model_path", "models/best.pt")
            if not Path(model_path).exists():
                model_path = "yolov8n.pt"
            model = YOLO(model_path)

            # Giữ resolution gốc (không resize xuống 640) để model thấy chi tiết tốt hơn.
            # Dùng conf=0.10 (thấp hơn video 0.25) vì ảnh đơn không lo track fragmentation.
            results = model(
                frame,
                conf=0.12,
                iou=0.35,   # tighter NMS: loại box chồng nhau tốt hơn
                device=cfg.get("device", "cpu"),
                verbose=False,
            )
            detections = results[0].boxes
            count_result = len(detections) if detections is not None else 0
            confidences = detections.conf.tolist() if detections is not None and len(detections) > 0 else []
            avg_conf = round(float(np.mean(confidences)), 2) if confidences else 0.0
            method_used = "yolo-single-frame"

            if annotated:
                # Vẽ bounding box lên ảnh
                annotated_frame = results[0].plot()
                label = f"Count: {count_result}  |  Conf: {avg_conf}"
                cv2.putText(annotated_frame, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        else:
            # ── Classical CV single-frame ────────────────────────────────
            tiny_config = build_tiny_config(base_config)
            c_cfg = tiny_config["classical"]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (c_cfg["blur_kernel"], c_cfg["blur_kernel"]), 0)
            block = c_cfg["adaptive_block_size"]
            thresh = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block, c_cfg["adaptive_c"],
            )
            k = c_cfg["morph_kernel_size"]
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid = [c for c in contours
                     if c_cfg["min_contour_area"] <= cv2.contourArea(c) <= c_cfg["max_contour_area"]]
            count_result = len(valid)
            avg_conf = 1.0
            method_used = "classical-cv-single-frame"

            if annotated:
                annotated_frame = frame.copy()
                cv2.drawContours(annotated_frame, valid, -1, (0, 255, 0), 2)
                label = f"Count: {count_result}"
                cv2.putText(annotated_frame, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        elapsed = round(time.time() - start_time, 2)

        # Trả về ảnh annotated nếu được yêu cầu
        if annotated:
            _, buf = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return StreamingResponse(
                io.BytesIO(buf.tobytes()),
                media_type="image/jpeg",
                headers={"X-Count-Result": str(count_result),
                         "X-Confidence-Score": str(avg_conf),
                         "X-Processing-Time": str(elapsed)},
            )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "filename": image.filename,
                "shrimp_type": shrimp_type,
                "method": method_used,
                "count_result": count_result,
                "confidence_score": avg_conf,
                "processing_time_seconds": elapsed,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
