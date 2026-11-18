"""End-to-end distillation pipeline CLI.

Runs: teacher eval → KD training → PTQ → QAT → benchmark table → ONNX export.

Usage
-----
    python scripts/run_distillation.py \
        --teacher yolov8s.pt \
        --student yolov8n.pt \
        --data data/pest.yaml \
        --epochs 50 \
        --device cpu \
        --out runs/distill_pest

The script writes the following to --out:
    best_student.pt          distilled FP32 student
    student_ptq_int8.pt      PTQ INT8 model
    student_qat_int8.pt      QAT INT8 model
    student_int8.onnx        ONNX export of QAT model
    benchmarks/results.csv   benchmark table
    train_log.csv            per-batch loss log
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root without install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from ultralytics import YOLO

from distill import (
    DistillationTrainer,
    PTQQuantizer,
    QATQuantizer,
    export_onnx,
    BenchmarkRunner,
)


# ---------------------------------------------------------------------------
# YOLOv8 channel widths at P3/P4/P5 for each size variant
# ---------------------------------------------------------------------------
YOLO_CHANNELS = {
    "n": [256, 256, 256],
    "s": [512, 512, 512],
    "m": [512, 512, 512],
    "l": [512, 512, 512],
    "x": [512, 512, 512],
}


def _infer_size(weights_path: str) -> str:
    """Guess YOLOv8 size from filename, e.g. yolov8s.pt → 's'."""
    name = Path(weights_path).stem.lower()
    for s in ("n", "s", "m", "l", "x"):
        if name.endswith(s):
            return s
    return "s"


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 distillation pipeline")
    p.add_argument("--teacher", required=True,
                   help="Path to teacher weights, e.g. yolov8s.pt")
    p.add_argument("--student", required=True,
                   help="Path to student weights, e.g. yolov8n.pt")
    p.add_argument("--data", required=True,
                   help="Path to YOLO dataset YAML, e.g. data/pest.yaml")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda-det", type=float, default=1.0,
                   help="Weight for detection loss")
    p.add_argument("--lambda-kd", type=float, default=0.5,
                   help="Weight for KD loss")
    p.add_argument("--temperature", type=float, default=4.0,
                   help="Softmax temperature for soft-label KD")
    p.add_argument("--device", default="cpu",
                   help="'cpu' or 'cuda' or '0'")
    p.add_argument("--out", default="runs/distill",
                   help="Output directory")
    p.add_argument("--skip-qat", action="store_true",
                   help="Skip QAT (faster, less accurate)")
    p.add_argument("--export-onnx", action="store_true",
                   help="Export final model to ONNX")
    return p.parse_args()


def build_loader(data_yaml: str, split: str = "val", batch: int = 8):
    """Build a DataLoader from a YOLO dataset YAML using Ultralytics."""
    from ultralytics.data.build import build_dataloader
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG

    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = data_yaml
    cfg.batch = batch
    cfg.imgsz = 640
    loader, _ = build_dataloader(cfg, split)
    return loader


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if args.device != "cuda" else "cuda:0"
    if device.isdigit():
        device = f"cuda:{device}"

    print("=" * 60)
    print("YOLOv8 Edge Distillation Pipeline")
    print(f"  teacher : {args.teacher}")
    print(f"  student : {args.student}")
    print(f"  data    : {args.data}")
    print(f"  epochs  : {args.epochs}")
    print(f"  device  : {device}")
    print("=" * 60)

    # Load models
    teacher_yolo = YOLO(args.teacher)
    student_yolo = YOLO(args.student)

    teacher_size = _infer_size(args.teacher)
    student_size = _infer_size(args.student)
    teacher_channels = YOLO_CHANNELS.get(teacher_size, [512, 512, 512])
    student_channels = YOLO_CHANNELS.get(student_size, [256, 256, 256])

    print(f"\nTeacher ({teacher_size}): channels {teacher_channels}")
    print(f"Student ({student_size}): channels {student_channels}")

    # Build data loaders
    print("\nBuilding data loaders...")
    try:
        train_loader = build_loader(args.data, split="train")
        val_loader = build_loader(args.data, split="val")
    except Exception as e:
        print(f"[warn] Could not build loaders via Ultralytics ({e}). "
              f"Using dummy loaders for structure demo.")
        train_loader = None
        val_loader = None

    # ----------------------------------------------------------------
    # Step 1: Knowledge Distillation Training
    # ----------------------------------------------------------------
    print("\n── Step 1: Knowledge Distillation ─────────────────────")
    trainer = DistillationTrainer(
        teacher=teacher_yolo.model,
        student=student_yolo.model,
        teacher_channels=teacher_channels,
        student_channels=student_channels,
        lambda_det=args.lambda_det,
        lambda_kd=args.lambda_kd,
        temperature=args.temperature,
        device=device,
    )

    if train_loader is not None:
        best_student_path = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            save_dir=out_dir,
        )
    else:
        # Save untrained student as placeholder
        best_student_path = out_dir / "best_student.pt"
        torch.save(student_yolo.model.state_dict(), best_student_path)
        print(f"[demo] Saved student → {best_student_path}")

    # Reload best student
    student_yolo.model.load_state_dict(
        torch.load(best_student_path, map_location=device)
    )
    student_fp32 = student_yolo.model

    # ----------------------------------------------------------------
    # Step 2: PTQ
    # ----------------------------------------------------------------
    print("\n── Step 2: Post-Training Quantization (INT8) ──────────")
    ptq = PTQQuantizer(student_fp32, backend="fbgemm")
    if val_loader is not None:
        ptq.calibrate(val_loader, n_batches=50, device=device)
        ptq_model = ptq.quantize()
        torch.save(ptq_model.state_dict(), out_dir / "student_ptq_int8.pt")
        print(f"PTQ INT8 model saved → {out_dir / 'student_ptq_int8.pt'}")
    else:
        ptq_model = student_fp32
        print("[demo] No val loader — skipping PTQ calibration")

    # ----------------------------------------------------------------
    # Step 3: QAT (optional)
    # ----------------------------------------------------------------
    if not args.skip_qat:
        print("\n── Step 3: Quantization-Aware Training (INT8) ─────────")
        qat = QATQuantizer(student_fp32, backend="fbgemm")
        qat_model = qat.prepare()

        if train_loader is not None:
            # Short QAT fine-tune: 10 epochs at 10× lower LR
            qat_trainer = DistillationTrainer(
                teacher=teacher_yolo.model,
                student=qat_model,
                teacher_channels=teacher_channels,
                student_channels=student_channels,
                lambda_det=args.lambda_det,
                lambda_kd=args.lambda_kd,
                device=device,
            )
            qat_trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=10,
                lr=args.lr / 10,
                save_dir=out_dir / "qat",
            )

        qat_int8 = qat.convert()
        torch.save(qat_int8.state_dict(), out_dir / "student_qat_int8.pt")
        print(f"QAT INT8 model saved → {out_dir / 'student_qat_int8.pt'}")
    else:
        qat_int8 = ptq_model

    # ----------------------------------------------------------------
    # Step 4: Benchmarking
    # ----------------------------------------------------------------
    print("\n── Step 4: Benchmark Table ─────────────────────────────")
    runner = BenchmarkRunner(
        val_loader=val_loader,
        class_names=["class0"],   # will be overridden per dataset
        device=device,
    )
    runner.add("teacher-fp32", teacher_yolo.model,
               weights_path=args.teacher)
    runner.add("student-fp32", student_fp32,
               weights_path=best_student_path)
    runner.add("student-ptq-int8", ptq_model)
    if not args.skip_qat:
        runner.add("student-qat-int8", qat_int8)

    results = runner.run()
    runner.print_table(results)
    runner.save_csv(results, out_dir / "benchmarks" / "results.csv")

    # ----------------------------------------------------------------
    # Step 5: ONNX export
    # ----------------------------------------------------------------
    if args.export_onnx:
        print("\n── Step 5: ONNX Export ─────────────────────────────────")
        export_onnx(
            qat_int8,
            out_dir / "student_int8.onnx",
            input_shape=(1, 3, 640, 640),
            opset=17,
            device=device,
        )

    print("\n✓ Pipeline complete. All outputs in:", out_dir)


if __name__ == "__main__":
    main()
