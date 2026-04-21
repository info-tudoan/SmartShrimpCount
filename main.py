"""
SmartShrimp Count — PoC entry point

Usage:
  # Classical CV (works immediately, no trained model needed)
  python main.py --video path/to/shrimp.mp4

  # YOLO + ByteTrack (needs models/best.pt from training)
  python main.py --video path/to/shrimp.mp4 --method yolo

  # Show live preview
  python main.py --video path/to/shrimp.mp4 --preview

  # Skip saving annotated video (faster)
  python main.py --video path/to/shrimp.mp4 --no-output-video
"""
import argparse
import sys
import yaml
from pathlib import Path


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="SmartShrimp Count — AI shrimp counting PoC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument(
        "--method",
        choices=["classical", "yolo"],
        default="classical",
        help="Counting method (default: classical)",
    )
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument("--output", help="Output annotated video path (auto-named if omitted)")
    parser.add_argument("--no-output-video", action="store_true", help="Skip saving annotated video")
    parser.add_argument("--preview", action="store_true", help="Show live preview window (press Q to quit)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video not found: {args.video}")
        sys.exit(1)

    config = load_config(args.config)
    output_dir = config["video"].get("output_dir", "outputs")

    output_video = None
    if not args.no_output_video:
        output_video = args.output or f"{output_dir}/{video_path.stem}_{args.method}_annotated.mp4"

    print(f"\nSmartShrimp Count PoC")
    print(f"  Video  : {video_path}")
    print(f"  Method : {args.method}")
    print()

    if args.method == "classical":
        from src.classical_counter import count_shrimp_classical
        result = count_shrimp_classical(
            video_path, config,
            output_path=output_video,
            show_preview=args.preview,
        )
    else:
        from src.yolo_counter import count_shrimp_yolo
        result = count_shrimp_yolo(
            video_path, config,
            output_path=output_video,
            show_preview=args.preview,
        )

    from src.report import generate_report
    generate_report(result, output_dir=output_dir)


if __name__ == "__main__":
    main()
