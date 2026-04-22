"""
SmartShrimp Count — REST API
POST /count  ->  Upload video, get shrimp count back

Usage:
  uvicorn api:app --host 0.0.0.0 --port 8000

Examples:
  # Large juvenile shrimp (De Heus WhatsApp video) -> dùng shrimp_type=large
  curl -X POST http://localhost:8000/count \
       -F "video=@whatsapp.mp4" \
       -F "shrimp_type=large"

  # Tiny post-larvae shrimp -> dùng shrimp_type=tiny
  curl -X POST http://localhost:8000/count \
       -F "video=@tomnho.mp4" \
       -F "shrimp_type=tiny"
"""
import shutil
import tempfile
import time
import yaml
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(
    title="SmartShrimp Count API",
    description="""
## AI-powered shrimp counting for De Heus Vietnam

### Chọn đúng `shrimp_type` theo loại tôm:

| shrimp_type | Loại tôm | Phương pháp | Thời gian |
|-------------|----------|-------------|-----------|
| `large` | Tôm giống lớn (>20px) | YOLO + ByteTrack | ~3–5 phút |
| `tiny` | Tôm post-larvae nhỏ (<10px) | Classical CV | ~2 phút |
""",
    version="1.1.0",
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

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "filename": video.filename,
                "shrimp_type": shrimp_type,
                "method": method_used,
                "estimated_count": result["estimated_count"],
                "avg_visible_per_frame": result.get("avg_visible", 0),
                "max_visible_per_frame": result.get("max_visible", 0),
                "frames_processed": result.get("frames_processed", 0),
                "processing_time_seconds": elapsed,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
