"""yolov8-edge-distillation — knowledge distillation + quantization for YOLOv8.

Compresses a YOLOv8s teacher into a YOLOv8n student via:
  - Feature-map knowledge distillation (P3/P4/P5 backbone stages)
  - Soft-label KD (Hinton 2015)
  - Post-Training Quantization (PTQ, INT8)
  - Quantization-Aware Training (QAT, INT8)
  - ONNX export for TensorRT / edge NPU deployment
  - Full benchmarking harness (mAP × latency × size)
"""

from .knowledge_distillation import KnowledgeDistiller, FeatureAdaptor
from .quantization import PTQQuantizer, QATQuantizer, export_onnx, benchmark_latency
from .trainer import DistillationTrainer
from .benchmark import BenchmarkRunner, BenchmarkResult

__all__ = [
    "KnowledgeDistiller",
    "FeatureAdaptor",
    "PTQQuantizer",
    "QATQuantizer",
    "export_onnx",
    "benchmark_latency",
    "DistillationTrainer",
    "BenchmarkRunner",
    "BenchmarkResult",
]
