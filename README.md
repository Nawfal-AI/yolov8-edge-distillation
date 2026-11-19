# YOLOv8 Edge Distillation Pipeline

[![CI](https://github.com/Nawfal-AI/yolov8-edge-distillation/actions/workflows/ci.yml/badge.svg)](https://github.com/Nawfal-AI/yolov8-edge-distillation/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Compresses a **YOLOv8s teacher** into a **YOLOv8n student** via knowledge
distillation, then exports to ONNX for deployment on constrained edge
hardware (Jetson, custom NPUs, embedded systems).

---

## Why this exists

Large foundation detection models are accurate but cannot run at real-time
rates on embedded platforms. This pipeline distills those models into
task-specific, hardware-friendly variants without retraining from scratch —
the same approach used for deploying vision models on autonomous systems
where latency and power budgets are hard constraints.

---

## Benchmark Results

Evaluated on a Tesla T4 GPU. Dataset: 2-class pest detection
(**Aspilanta** + **eu\_moth**), 300 validation images, 1,713 instances.

| Model | mAP@50 | mAP@50-95 | Latency (ms) | FPS | Size (MB) | Compression |
|---|---|---|---|---|---|---|
| Teacher (YOLOv8s) | **0.9727** | 0.7753 | 17.09 | 58.5 | 21.47 | 1.0× |
| Student (YOLOv8n) | **0.9615** | 0.7366 | **8.27** | **120.9** | **6.2** | **3.46×** |

> The student is **3.46× smaller** and runs at **2.07× the teacher's speed**
> (120.9 FPS vs 58.5 FPS), retaining **98.8% of the teacher's mAP@50**.

Per-class breakdown (teacher, validation set):

| Class | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|
| Aspilanta | 287 | 1,414 | 0.968 | 0.938 | 0.975 | 0.746 |
| eu\_moth | 154 | 299 | 0.984 | 0.943 | 0.970 | 0.805 |

---

## Pretrained Weights

| File | Description | mAP@50 | Size |
|---|---|---|---|
| [`weights/student_best.pt`](weights/student_best.pt) | Distilled YOLOv8n student | 0.9615 | 6.3 MB |
| [`weights/student_best.onnx`](weights/student_best.onnx) | ONNX export (opset 18) | 0.9615 | 13 MB |

Download from the [Releases](https://github.com/Nawfal-AI/yolov8-edge-distillation/releases) page
if Git LFS is not available.

**Run inference with the student:**

```python
from ultralytics import YOLO

model = YOLO("weights/student_best.pt")
results = model("your_image.jpg")
results[0].show()
```

**Run with ONNX (edge deployment):**

```python
model = YOLO("weights/student_best.onnx", task="detect")
results = model("your_image.jpg")
```

---

## Method

### 1. Knowledge Distillation

Feature-map distillation at P3/P4/P5 backbone stages combined with
Hinton-style soft-label KD:

```
L_total = λ_det · L_detection + λ_kd · (β · L_feat + α · L_soft)
```

- **L\_feat** — MSE between L2-normalised teacher/student feature maps,
  bridged by a 1×1 `FeatureAdaptor` to handle the channel-width gap
  (teacher: 512 ch, student: 256 ch at P3/P4/P5)
- **L\_soft** — KL-divergence on temperature-scaled logits (T = 4.0, Hinton 2015)
- Teacher runs fully frozen (eval + no-grad); only student and adaptors update

### 2. Post-Training Quantization (PTQ)

Calibrate per-channel scale/zero-point from 50 validation batches → static
INT8 conversion via `torch.quantization`. No retraining required.

### 3. ONNX Export

Student exported to ONNX opset 18 via Ultralytics, slimmed with `onnxslim`,
ready for TensorRT / OpenVINO / edge NPU runtimes.

---

## Quick Start

```bash
pip install -e .
```

**Train teacher + distill student:**

```bash
python scripts/run_distillation.py \
    --teacher yolov8s.pt \
    --student yolov8n.pt \
    --data configs/pest_detection.yaml \
    --epochs 30 \
    --skip-qat \
    --export-onnx \
    --out runs/distill_pest
```

**Benchmark saved checkpoints:**

```bash
python scripts/run_benchmark.py \
    --models teacher:weights/teacher_best.pt \
             student:weights/student_best.pt \
    --data configs/pest_detection.yaml
```

**Run unit tests (no data or weights needed):**

```bash
pytest tests/ -v
```

---

## Project Structure

```
yolov8-edge-distillation/
├── src/distill/
│   ├── knowledge_distillation.py  # Feature-map KD + soft-label KD losses
│   ├── quantization.py            # PTQ, ONNX export, latency benchmark
│   ├── trainer.py                 # Training loop with KD loss integration
│   └── benchmark.py               # mAP × latency × size comparison table
├── scripts/
│   ├── run_distillation.py        # End-to-end CLI pipeline
│   └── run_benchmark.py           # Standalone benchmark CLI
├── configs/
│   └── pest_detection.yaml        # Dataset + training config
├── weights/
│   ├── student_best.pt            # Distilled student weights
│   ├── student_best.onnx          # ONNX export for edge deployment
│   └── README.md                  # Weight download instructions
└── tests/
    └── test_distillation.py       # Unit tests
```

---

## Dataset

| Split | Images | Instances |
|---|---|---|
| Train | ~1,712 | ~9,700 |
| Val | 300 | 1,713 |

Classes: **Aspilanta** (leafmining moth, Lepidoptera: Heliozelidae) · **eu\_moth**

The dataset was developed as part of the MSc thesis
[aspilanta-fewshot-detection](https://github.com/Nawfal-AI/aspilanta-fewshot-detection).

---

## Connection to Thesis

This project extends my MSc thesis few-shot pest detector
([aspilanta-fewshot-detection](https://github.com/Nawfal-AI/aspilanta-fewshot-detection),
mAP@50 = 0.979 from only 44 real training images) by addressing the
deployment gap: the thesis model is accurate but too large for real-time
inference on constrained field hardware. This pipeline closes that gap —
the distilled student runs at **120.9 FPS** on a T4 GPU and exports cleanly
to ONNX for edge NPU deployment, with no meaningful accuracy loss.

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Ultralytics 8.1+
- torchmetrics
- onnx + onnxruntime

```bash
pip install -r requirements.txt
```
