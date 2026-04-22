"""
Quick training status check.
Usage: python scripts/check_training.py
"""
import csv
from pathlib import Path


def find_latest_run():
    base = Path("runs/detect/runs/train")
    if not base.exists():
        base = Path("runs/train")
    if not base.exists():
        return None
    runs = sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for r in runs:
        if (r / "results.csv").exists():
            return r
    return None


def show_status():
    run = find_latest_run()
    if not run:
        print("No training run found.")
        return

    print(f"Latest run: {run.name}")
    csv_path = run / "results.csv"
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("No epochs completed yet.")
        return

    last = rows[-1]
    epoch = last["epoch"].strip()
    map50 = float(last["metrics/mAP50(B)"].strip())
    map50_95 = float(last["metrics/mAP50-95(B)"].strip())
    precision = float(last["metrics/precision(B)"].strip())
    recall = float(last["metrics/recall(B)"].strip())

    # Best mAP50 so far
    best = max(rows, key=lambda r: float(r["metrics/mAP50(B)"].strip()))
    best_epoch = best["epoch"].strip()
    best_map50 = float(best["metrics/mAP50(B)"].strip())

    print(f"  Epochs completed : {epoch}")
    print(f"  Current mAP50    : {map50:.4f}  ({map50*100:.1f}%)")
    print(f"  Current P / R    : {precision:.3f} / {recall:.3f}")
    print(f"  Best mAP50       : {best_map50:.4f} at epoch {best_epoch}")
    print(f"  mAP50-95         : {map50_95:.4f}")

    weights = run / "weights" / "best.pt"
    if weights.exists():
        mb = weights.stat().st_size / 1e6
        print(f"  best.pt          : {weights}  ({mb:.1f} MB)")

    # Progress bar
    try:
        current = int(epoch)
        total = 150
        bar_len = 30
        filled = int(bar_len * current / total)
        bar = "#" * filled + "-" * (bar_len - filled)
        pct = current / total * 100
        print(f"\n  Progress: [{bar}] {current}/{total} ({pct:.0f}%)")
    except Exception:
        pass


if __name__ == "__main__":
    show_status()
