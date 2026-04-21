"""
SmartShrimp Count — REST API
POST /count  ->  Upload video, get shrimp count back

Usage:
  uvicorn api:app --host 0.0.0.0 --port 8000

Test with curl:
  curl -X POST http://localhost:8000/count \
       -F "video=@data/videos/tomnho.mp4" \
       -F "method=classical"
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
    description="AI-powered shrimp counting from video for De Heus Vietnam",
    version="1.0.0",
)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@app.get("/")
def root():
    return {
        "service": "SmartShrimp Count API",
        "version": "1.0.0",
        "endpoints": {
            "POST /count": "Upload video, returns shrimp count",
            "GET /health": "Health check",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/count")
async def count_shrimp(
    video: UploadFile = File(..., description="Video file (mp4, avi, mov)"),
    method: str = Form(
        default="classical",
        description="Counting method: 'classical' (best for tiny post-larvae) or 'yolo' (best for large shrimp)",
    ),
):
    """
    Count shrimp in an uploaded video.

    - **video**: Video file (mp4, avi, mov, mkv)
    - **method**: `classical` for tiny post-larvae shrimp, `yolo` for large juvenile shrimp

    Returns estimated shrimp count with detailed statistics.
    """

    # Validate method
    if method not in ("classical", "yolo"):
        raise HTTPException(
            status_code=400,
            detail="method must be 'classical' or 'yolo'",
        )

    # Validate file type
    allowed_ext = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
    suffix = Path(video.filename or "video.mp4").suffix.lower()
    if suffix not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(allowed_ext)}",
        )

    # Save uploaded file to temp location
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_video = tmp_dir / f"upload{suffix}"

    try:
        with open(tmp_video, "wb") as f:
            shutil.copyfileobj(video.file, f)

        config = load_config()
        start_time = time.time()

        if method == "classical":
            from src.classical_counter import count_shrimp_classical
            result = count_shrimp_classical(
                tmp_video,
                config,
                output_path=None,   # no video output for API
                show_preview=False,
            )
        else:
            from src.yolo_counter import count_shrimp_yolo
            result = count_shrimp_yolo(
                tmp_video,
                config,
                output_path=None,
                show_preview=False,
            )

        elapsed = round(time.time() - start_time, 1)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "filename": video.filename,
                "method": method,
                "estimated_count": result["estimated_count"],
                "avg_visible_per_frame": result.get("avg_visible", 0),
                "max_visible_per_frame": result.get("max_visible", 0),
                "frames_processed": result.get("frames_processed", 0),
                "processing_time_seconds": elapsed,
                "model_used": result.get("model_used", "classical-cv"),
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temp files
        shutil.rmtree(tmp_dir, ignore_errors=True)
