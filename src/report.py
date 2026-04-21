import json
from datetime import datetime
from pathlib import Path


def generate_report(result, output_dir="outputs"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_stem = Path(result["video_path"]).stem
    report_path = output_dir / f"{video_stem}_{result['method']}_{ts}.json"

    report = {
        "timestamp": ts,
        "video": result["video_path"],
        "method": result["method"],
        "estimated_shrimp_count": result["estimated_count"],
        "details": {
            "max_visible_per_frame": result.get("max_visible"),
            "avg_visible_per_frame": result.get("avg_visible"),
            "frames_processed": result.get("frames_processed"),
            "total_frames": result.get("total_frames"),
        },
    }

    if result["method"] == "classical":
        report["details"]["total_tracked_ids"] = result.get("total_tracked_ids")
        report["note"] = (
            "estimated_count = 90th percentile of visible shrimp per frame. "
            "total_tracked_ids may be inflated if the same shrimp was lost and re-registered."
        )
    elif result["method"] == "yolo":
        report["details"]["unique_track_ids"] = result.get("unique_track_ids")
        report["details"]["model_used"] = result.get("model_used")

    if result.get("output_video"):
        report["output_video"] = result["output_video"]

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    _print_summary(result, report_path)
    return report_path


def _print_summary(result, report_path):
    w = 52
    print()
    print("=" * w)
    print("  SMART SHRIMP COUNT — RESULT")
    print("=" * w)
    print(f"  Video    : {Path(result['video_path']).name}")
    print(f"  Method   : {result['method'].upper()}")
    print(f"  Estimated: {result['estimated_count']:,} shrimp")
    print(f"  Max visible / frame : {result.get('max_visible', 'N/A')}")
    print(f"  Avg visible / frame : {result.get('avg_visible', 'N/A')}")
    print(f"  Frames processed    : {result.get('frames_processed', 'N/A')}")
    if result["method"] == "classical":
        print(f"  Total tracked IDs   : {result.get('total_tracked_ids', 'N/A')}")
    elif result["method"] == "yolo":
        print(f"  Unique track IDs    : {result.get('unique_track_ids', 'N/A')}")
    print("-" * w)
    print(f"  Report   : {report_path}")
    if result.get("output_video"):
        print(f"  Video    : {result['output_video']}")
    print("=" * w)
    print()
