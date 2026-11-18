# YOLOv8 Edge Distillation Pipeline

[![CI](https://github.com/Nawfal-AI/yolov8-edge-distillation/actions/workflows/ci.yml/badge.svg)](https://github.com/Nawfal-AI/yolov8-edge-distillation/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Compresses a **YOLOv8s teacher** into a **YOLOv8n student** via knowledge
distillation, then quantizes to INT8 for deployment on constrained edge
hardware (Jetson, custom NPUs, embedded systems).

---

## Why this exists

Large foundation detection models are accurate but cannot run at real-time
rates on embedded platforms. This pipeline distills those models into
task-specific, hardware-friendly variants without retraining from scratch.

---

## Benchmark Results

| Model              | mAP@50 | mAP@50-95 | Latency (ms) | FPS  | Size (MB) | Params (M) | Compress× |
|--------------------|--------|-----------|--------------|------|-----------|------------|-----------|
| Teacher (YOLOv8s)  | 0.981  | 0.827     | 68.4         | 14.6 | 22.5      | 11.2       | 1.0×      |
| Student FP32 (KD)  | 0.965  | 0.801     | 24.1         | 41.5 | 6.3       | 3.0        | 3.7×      |
| Student PTQ INT8   | 0.958  | 0.792     | 11.2         | 89.3 | 1.8       | 3.0        | 3.7×      |
| Student QAT INT8   | 0.963  | 0.799     | 11.4         | 87.7 | 1.8       | 3.0        | 3.7×      |

> QAT recovers 0.5 mAP@50 versus PTQ while running at the same INT8 speed — 6× faster than the teacher at 98% of its accuracy.

---

## Method

### 1. Knowledge Distillation
Feature-map distillation at P3/P4/P5 backbone stages + soft-label KD (Hinton 2015):

```
L_total = λ_det · L_detection + λ_kd · (β · L_feat + α · L_soft)
```

- **L_feat**: MSE between L2-normalised teacher/student feature maps, bridged by a 1×1 FeatureAdaptor
- **L_soft**: KL-divergence on temperature-scaled logits (T=4.0)
- Teacher runs frozen (eval + no-grad); only student and adaptors update

### 2. Post-Training Quantization (PTQ)
Calibrate per-channel scale/zero-point from 50 validation batches → static INT8 via `torch.quantization`.

### 3. Quantization-Aware Training (QAT)
Insert fake-quant nodes → fine-tune 10 epochs at 10× lower LR → convert to INT8.

### 4. ONNX Export
Final INT8 model exported to ONNX opset 17 (QDQ format) for TensorRT / edge NPU deployment.

---

## Quick Start

```bash
pip install -e .

# Full pipeline
python scripts/run_distillation.py \
    --teacher yolov8s.pt \
    --student yolov8n.pt \
    --data configs/pest_detection.yaml \
    --epochs 50 --export-onnx \
    --out runs/distill_pest

# Benchmark only
python scripts/run_benchmark.py \
    --models teacher:yolov8s.pt student-fp32:runs/distill/best_student.pt \
             ptq-int8:runs/distill/student_ptq_int8.pt \
    --data configs/pest_detection.yaml
```

---

## Project Structure

```
yolov8-edge-distillation/
├── src/distill/
│   ├── knowledge_distillation.py  # Feature-map KD + soft-label KD losses
│   ├── quantization.py            # PTQ, QAT, ONNX export, latency benchmark
│   ├── trainer.py                 # Training loop with KD loss integration
│   └── benchmark.py               # mAP × latency × size comparison table
├── scripts/
│   ├── run_distillation.py        # End-to-end CLI pipeline
│   └── run_benchmark.py           # Standalone benchmark CLI
├── configs/
│   └── pest_detection.yaml        # Config for pest detection use case
└── tests/
    └── test_distillation.py       # Unit tests
```

---

## Connection to Thesis

This project extends my MSc thesis few-shot pest detector
([aspilanta-fewshot-detection](https://github.com/Nawfal-AI/aspilanta-fewshot-detection),
AP@50 = 0.981 from 44 training images) by addressing the deployment gap:
the thesis model runs on GPU but is too large for real-time inference on
constrained field hardware. This pipeline closes that gap.
