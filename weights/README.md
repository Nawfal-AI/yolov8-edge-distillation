# Pretrained Weights

Trained on the pest detection dataset (Aspilanta + eu\_moth),
30 epochs, Tesla T4 GPU.

| File | Description | mAP@50 | mAP@50-95 | FPS | Size |
|---|---|---|---|---|---|
| `student_best.pt` | Distilled YOLOv8n student | 0.9615 | 0.7366 | 120.9 | 6.3 MB |
| `student_best.onnx` | ONNX export of student (opset 18) | 0.9615 | 0.7366 | 120.9 | 13 MB |

The teacher weights are not included here — they can be reproduced by
fine-tuning `yolov8s.pt` on the dataset for 30 epochs, or downloaded
from the [Releases](https://github.com/Nawfal-AI/yolov8-edge-distillation/releases) page.

---

## Usage

**Inference with PyTorch weights:**

```python
from ultralytics import YOLO

model = YOLO("weights/student_best.pt")
results = model("your_image.jpg", conf=0.25)
results[0].show()
```

**Inference with ONNX (edge/embedded deployment):**

```python
from ultralytics import YOLO

model = YOLO("weights/student_best.onnx", task="detect")
results = model("your_image.jpg", conf=0.25)
```

**Validate on your own dataset:**

```python
from ultralytics import YOLO

model = YOLO("weights/student_best.pt")
metrics = model.val(data="your_dataset.yaml")
print(f"mAP@50: {metrics.box.map50:.4f}")
print(f"mAP@50-95: {metrics.box.map:.4f}")
```

---

## Note on Git LFS

`.pt` and `.onnx` files are large binary files. If they are missing after
cloning, either:

1. Run `git lfs pull` (if Git LFS is configured), or
2. Download directly from the
   [Releases](https://github.com/Nawfal-AI/yolov8-edge-distillation/releases) page
