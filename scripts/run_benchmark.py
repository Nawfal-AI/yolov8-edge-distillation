"""Standalone benchmark script — compare any set of saved model checkpoints.

Usage
-----
    python scripts/run_benchmark.py \
        --models teacher:yolov8s.pt student-fp32:runs/distill/best_student.pt \
                 ptq-int8:runs/distill/student_ptq_int8.pt \
        --data data/pest.yaml \
        --device cpu

Outputs a table to stdout and saves benchmarks/results.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from ultralytics import YOLO

from distill import BenchmarkRunner


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True,
                   metavar="NAME:PATH",
                   help="Space-separated name:weights_path pairs")
    p.add_argument("--data", default=None,
                   help="YOLO dataset YAML for mAP evaluation")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="benchmarks/results.csv")
    return p.parse_args()


def main():
    args = parse_args()

    entries = []
    for token in args.models:
        if ":" not in token:
            print(f"[error] expected NAME:PATH, got '{token}'")
            sys.exit(1)
        name, path = token.split(":", 1)
        entries.append((name, path))

    # Build val loader
    val_loader = None
    if args.data:
        try:
            from ultralytics.data.build import build_dataloader
            from ultralytics.cfg import get_cfg
            from ultralytics.utils import DEFAULT_CFG
            cfg = get_cfg(DEFAULT_CFG)
            cfg.data = args.data
            cfg.batch = 8
            cfg.imgsz = 640
            val_loader, _ = build_dataloader(cfg, "val")
        except Exception as e:
            print(f"[warn] Could not build val loader: {e}")

    runner = BenchmarkRunner(
        val_loader=val_loader,
        class_names=["class0"],
        device=args.device,
    )

    for name, weights_path in entries:
        yolo = YOLO(weights_path)
        runner.add(name, yolo.model, weights_path=weights_path)

    results = runner.run()
    runner.print_table(results)
    runner.save_csv(results, args.out)


if __name__ == "__main__":
    main()
