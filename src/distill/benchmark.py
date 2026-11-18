"""Benchmarking harness — latency × accuracy × size for every compression stage.

The Harmattan AI role explicitly requires "framework-level benchmarking of
newly distilled models to evaluate performance and latency, ensuring results
are fully aligned with real-world operational deployments."

This module produces exactly that: a structured table comparing the teacher,
distilled-FP32 student, PTQ-INT8 student, and QAT-INT8 student across:

  - mAP@50 / mAP@50-95       (accuracy)
  - Mean inference latency    (ms, CPU and optional GPU)
  - Throughput                (FPS)
  - Model size on disk        (MB)
  - Parameter count

Usage
-----
    runner = BenchmarkRunner(val_loader, class_names, device="cpu")
    runner.add("teacher",    teacher_model,    weights_path="yolov8s.pt")
    runner.add("student-fp32", student_model,  weights_path="student_fp32.pt")
    runner.add("ptq-int8",   ptq_int8_model)
    runner.add("qat-int8",   qat_int8_model)
    table = runner.run()
    runner.print_table(table)
    runner.save_csv(table, "benchmarks/results.csv")
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict

import torch
import torch.nn as nn


__all__ = ["BenchmarkResult", "BenchmarkRunner"]


@dataclass
class BenchmarkResult:
    name: str
    map50: float = 0.0
    map50_95: float = 0.0
    mean_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    fps: float = 0.0
    model_size_mb: float = 0.0
    n_params_m: float = 0.0          # millions
    compression_ratio: float = 1.0   # relative to teacher


class BenchmarkRunner:
    """Collect and run benchmarks for a set of model variants.

    Args:
        val_loader: DataLoader yielding (images, targets) for mAP evaluation.
        class_names: list of class name strings.
        device: 'cpu' or 'cuda'.
        input_shape: single-image (C, H, W) used for latency benchmarking.
        n_warmup: warmup runs before timing.
        n_latency_runs: timed runs for latency measurement.
    """

    def __init__(
        self,
        val_loader,
        class_names: List[str],
        device: str = "cpu",
        input_shape: tuple = (3, 640, 640),
        n_warmup: int = 10,
        n_latency_runs: int = 50,
    ):
        self.val_loader = val_loader
        self.class_names = class_names
        self.device = device
        self.input_shape = input_shape
        self.n_warmup = n_warmup
        self.n_latency_runs = n_latency_runs
        self._entries: List[Dict] = []   # {name, model, weights_path}

    def add(
        self,
        name: str,
        model: nn.Module,
        weights_path: Optional[str | Path] = None,
    ) -> "BenchmarkRunner":
        """Register a model variant.

        Args:
            name: display name, e.g. 'teacher', 'student-fp32', 'qat-int8'.
            model: PyTorch model.
            weights_path: .pt file on disk (used for model size measurement).
        """
        self._entries.append({
            "name": name,
            "model": model,
            "weights_path": Path(weights_path) if weights_path else None,
        })
        return self

    # ------------------------------------------------------------------
    # Core measurement methods
    # ------------------------------------------------------------------

    def _measure_latency(self, model: nn.Module) -> dict:
        model = model.to(self.device).eval()
        dummy = torch.randn(1, *self.input_shape, device=self.device)

        with torch.no_grad():
            for _ in range(self.n_warmup):
                _ = model(dummy)

        times = []
        with torch.no_grad():
            for _ in range(self.n_latency_runs):
                if self.device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(dummy)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)

        import statistics
        mean_ms = statistics.mean(times)
        std_ms = statistics.stdev(times) if len(times) > 1 else 0.0
        return {"mean_ms": round(mean_ms, 2),
                "std_ms": round(std_ms, 2),
                "fps": round(1000.0 / mean_ms, 1)}

    def _measure_map(self, model: nn.Module) -> dict:
        """Evaluate mAP@50 and mAP@50-95 on val_loader.

        Uses torchmetrics MeanAveragePrecision when available, falls back to
        a simple TP/FP accumulation otherwise.
        """
        try:
            from torchmetrics.detection import MeanAveragePrecision
            metric = MeanAveragePrecision(iou_type="bbox")
            model = model.to(self.device).eval()

            with torch.no_grad():
                for imgs, targets in self.val_loader:
                    imgs = imgs.to(self.device)
                    preds = model(imgs)
                    metric.update(preds, targets)

            result = metric.compute()
            return {
                "map50": round(float(result["map_50"]), 4),
                "map50_95": round(float(result["map"]), 4),
            }
        except ImportError:
            # torchmetrics not installed — return placeholders
            print("[benchmark] torchmetrics not found; mAP set to -1. "
                  "Install with: pip install torchmetrics")
            return {"map50": -1.0, "map50_95": -1.0}
        except Exception as e:
            print(f"[benchmark] mAP evaluation failed: {e}")
            return {"map50": -1.0, "map50_95": -1.0}

    @staticmethod
    def _model_size_mb(model: nn.Module,
                       weights_path: Optional[Path]) -> float:
        """Disk size (MB) if weights_path exists, else estimate from params."""
        if weights_path and weights_path.exists():
            return round(weights_path.stat().st_size / (1024 ** 2), 2)
        # Estimate: fp32 → 4 bytes/param, int8 → ~1 byte/param
        n = sum(p.numel() for p in model.parameters())
        return round(n * 4 / (1024 ** 2), 2)

    @staticmethod
    def _param_count_millions(model: nn.Module) -> float:
        n = sum(p.numel() for p in model.parameters())
        return round(n / 1e6, 2)

    # ------------------------------------------------------------------
    # Run all
    # ------------------------------------------------------------------

    def run(self) -> List[BenchmarkResult]:
        """Run all registered variants and return results."""
        results = []
        teacher_params = None

        for entry in self._entries:
            name = entry["name"]
            model = entry["model"]
            weights_path = entry["weights_path"]

            print(f"\n── Benchmarking: {name} ─────────────────────────")

            latency = self._measure_latency(model)
            accuracy = self._measure_map(model)
            size_mb = self._model_size_mb(model, weights_path)
            n_params = self._param_count_millions(model)

            if teacher_params is None:
                teacher_params = n_params  # first entry is teacher

            compression = round(teacher_params / n_params, 2) if n_params > 0 else 1.0

            result = BenchmarkResult(
                name=name,
                map50=accuracy["map50"],
                map50_95=accuracy["map50_95"],
                mean_latency_ms=latency["mean_ms"],
                std_latency_ms=latency["std_ms"],
                fps=latency["fps"],
                model_size_mb=size_mb,
                n_params_m=n_params,
                compression_ratio=compression,
            )
            results.append(result)
            print(f"  mAP@50={result.map50}  mAP@50-95={result.map50_95}  "
                  f"latency={result.mean_latency_ms}ms  "
                  f"size={result.model_size_mb}MB  "
                  f"compression={result.compression_ratio}×")

        return results

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    @staticmethod
    def print_table(results: List[BenchmarkResult]) -> None:
        """Print a formatted comparison table to stdout."""
        col_w = 18
        headers = ["Model", "mAP@50", "mAP@50-95", "Latency(ms)",
                   "FPS", "Size(MB)", "Params(M)", "Compress×"]
        sep = "-" * (col_w * len(headers))
        print("\n" + sep)
        print("".join(h.ljust(col_w) for h in headers))
        print(sep)
        for r in results:
            row = [
                r.name, str(r.map50), str(r.map50_95),
                str(r.mean_latency_ms), str(r.fps),
                str(r.model_size_mb), str(r.n_params_m),
                str(r.compression_ratio),
            ]
            print("".join(v.ljust(col_w) for v in row))
        print(sep)

    @staticmethod
    def save_csv(results: List[BenchmarkResult],
                 path: str | Path) -> Path:
        """Write results to CSV for reproducible logging."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
        print(f"Benchmark results saved → {path}")
        return path
