"""Quantization utilities — PTQ and QAT for YOLOv8 student models.

Covers the two quantization strategies the Harmattan AI role explicitly names:

* **Post-Training Quantization (PTQ)**: calibrate scale/zero-point from a
  representative dataset after training, no retraining required. Fast to apply
  but typically loses 1–3 mAP@50 points on detection heads.
* **Quantization-Aware Training (QAT)**: insert fake-quant nodes during
  fine-tuning so the model learns to tolerate quantization error. Recovers most
  of the PTQ accuracy gap at the cost of an extra training pass (~10 epochs).

Additionally provides ONNX export with dynamic/static INT8 for TensorRT-style
edge deployment.

Usage
-----
    # PTQ
    ptq = PTQQuantizer(model)
    ptq.calibrate(calibration_loader)
    int8_model = ptq.quantize()

    # QAT
    qat = QATQuantizer(model)
    qat.prepare()
    # ... fine-tune for a few epochs ...
    int8_model = qat.convert()

    # ONNX export
    export_onnx(int8_model, "student_int8.onnx", input_shape=(1,3,640,640))
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.quantization as tq


__all__ = [
    "PTQQuantizer",
    "QATQuantizer",
    "export_onnx",
    "benchmark_latency",
    "count_parameters",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def benchmark_latency(
    model: nn.Module,
    input_shape: tuple = (1, 3, 640, 640),
    n_warmup: int = 10,
    n_runs: int = 50,
    device: str = "cpu",
) -> dict:
    """Measure mean inference latency and throughput.

    Args:
        model: any PyTorch model.
        input_shape: (B, C, H, W).
        n_warmup: runs discarded before timing.
        n_runs: timed runs.
        device: 'cpu' or 'cuda'.

    Returns:
        dict with keys: mean_ms, std_ms, fps, device.
    """
    model = model.to(device).eval()
    dummy = torch.randn(*input_shape, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    import statistics
    mean_ms = statistics.mean(times)
    std_ms = statistics.stdev(times)
    fps = 1000.0 / mean_ms

    return {"mean_ms": round(mean_ms, 2),
            "std_ms": round(std_ms, 2),
            "fps": round(fps, 1),
            "device": device}


# ---------------------------------------------------------------------------
# PTQ
# ---------------------------------------------------------------------------

class PTQQuantizer:
    """Post-Training Quantization wrapper for a PyTorch model.

    Fuses Conv-BN-ReLU triplets, inserts observer modules, runs calibration,
    then converts to static INT8.

    Args:
        model: a trained FP32 model (not yet quantized).
        backend: 'fbgemm' (x86) or 'qnnpack' (ARM/edge).
    """

    def __init__(self, model: nn.Module, backend: str = "fbgemm"):
        self.model = copy.deepcopy(model).eval()
        self.backend = backend
        self._calibrated = False

    def calibrate(
        self,
        loader,
        n_batches: int = 50,
        device: str = "cpu",
        preprocess: Optional[Callable] = None,
    ) -> "PTQQuantizer":
        """Run calibration pass to collect activation statistics.

        Args:
            loader: DataLoader yielding (images, _) batches.
            n_batches: how many batches to use. 50 is typically sufficient.
            device: where to run calibration.
            preprocess: optional transform applied to each batch before forward.
        """
        torch.backends.quantized.engine = self.backend
        self.model.qconfig = tq.get_default_qconfig(self.backend)

        # Fuse Conv-BN-ReLU for efficiency
        try:
            self.model = tq.fuse_modules(self.model,
                                          _find_fusable_triplets(self.model))
        except Exception:
            pass  # fuse_modules is best-effort on custom architectures

        tq.prepare(self.model, inplace=True)
        self.model = self.model.to(device)

        with torch.no_grad():
            for i, (imgs, *_) in enumerate(loader):
                if i >= n_batches:
                    break
                imgs = imgs.to(device)
                if preprocess is not None:
                    imgs = preprocess(imgs)
                self.model(imgs)

        self._calibrated = True
        return self

    def quantize(self) -> nn.Module:
        """Convert calibrated model to static INT8."""
        if not self._calibrated:
            raise RuntimeError("Call .calibrate() before .quantize().")
        tq.convert(self.model, inplace=True)
        return self.model


# ---------------------------------------------------------------------------
# QAT
# ---------------------------------------------------------------------------

class QATQuantizer:
    """Quantization-Aware Training wrapper.

    Inserts fake-quant nodes so the model experiences quantization error
    during fine-tuning. After a short fine-tune (.convert()) returns a
    static INT8 model that is more accurate than PTQ.

    Typical recipe: 10–20 fine-tune epochs at 10× lower LR than original
    training, with the backbone frozen and only the head free.

    Args:
        model: a trained FP32 model.
        backend: 'fbgemm' (x86) or 'qnnpack' (ARM/edge).
    """

    def __init__(self, model: nn.Module, backend: str = "fbgemm"):
        self.model = copy.deepcopy(model).train()
        self.backend = backend

    def prepare(self) -> nn.Module:
        """Insert fake-quant observers. Call once before fine-tuning."""
        torch.backends.quantized.engine = self.backend
        self.model.qconfig = tq.get_default_qat_qconfig(self.backend)

        try:
            self.model = tq.fuse_modules(self.model,
                                          _find_fusable_triplets(self.model))
        except Exception:
            pass

        tq.prepare_qat(self.model, inplace=True)
        return self.model

    def convert(self) -> nn.Module:
        """Freeze fake-quant params and convert to static INT8.

        Call after fine-tuning is complete.
        """
        self.model.eval()
        tq.convert(self.model, inplace=True)
        return self.model


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple = (1, 3, 640, 640),
    opset: int = 17,
    dynamic_axes: Optional[dict] = None,
    device: str = "cpu",
) -> Path:
    """Export a PyTorch model (FP32 or INT8) to ONNX.

    Args:
        model: trained PyTorch model.
        output_path: where to write the .onnx file.
        input_shape: (B, C, H, W) — use batch=1 for static export.
        opset: ONNX opset version. 17 covers QDQ nodes for TensorRT.
        dynamic_axes: set to enable dynamic batch / spatial dims, e.g.:
            {"images": {0: "batch"}, "output": {0: "batch"}}
        device: export device.

    Returns:
        Path to written .onnx file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.to(device).eval()
    dummy = torch.randn(*input_shape, device=device)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=opset,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"ONNX export → {output_path}  ({size_mb:.1f} MB)")
    return output_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_fusable_triplets(model: nn.Module) -> list[list[str]]:
    """Walk the model and collect Conv → BN → ReLU name triples.

    ``torch.quantization.fuse_modules`` requires explicit layer name lists;
    this function discovers them automatically for sequential-style models.
    """
    triplets = []
    named = dict(model.named_modules())
    keys = list(named.keys())

    for i in range(len(keys) - 2):
        a, b, c = named[keys[i]], named[keys[i+1]], named[keys[i+2]]
        if (isinstance(a, (nn.Conv2d, nn.ConvTranspose2d))
                and isinstance(b, nn.BatchNorm2d)
                and isinstance(c, (nn.ReLU, nn.ReLU6))):
            triplets.append([keys[i], keys[i+1], keys[i+2]])

    return triplets
