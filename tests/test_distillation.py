"""Unit tests for the distillation and quantization modules.

Run with: pytest tests/ -v
"""

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Minimal toy models for testing without real weights
# ---------------------------------------------------------------------------

class ToyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 512, 3, padding=1)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()
        self.head = nn.Linear(512, 4)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = x.mean(dim=(2, 3))
        return self.head(x)


class ToyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 256, 3, padding=1)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.head = nn.Linear(256, 4)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = x.mean(dim=(2, 3))
        return self.head(x)


# ---------------------------------------------------------------------------
# KD tests
# ---------------------------------------------------------------------------

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from distill.knowledge_distillation import (
    KnowledgeDistiller,
    FeatureAdaptor,
    soft_label_loss,
    feature_map_loss,
)


def test_soft_label_loss_shape():
    logits_t = torch.randn(8, 4)
    logits_s = torch.randn(8, 4)
    loss = soft_label_loss(logits_t, logits_s, temperature=4.0)
    assert loss.shape == (), "soft_label_loss must return a scalar"
    assert loss.item() >= 0, "KL divergence must be non-negative"


def test_feature_map_loss_same_channels():
    t_feat = torch.randn(2, 512, 20, 20)
    s_feat = torch.randn(2, 512, 20, 20)
    loss = feature_map_loss(s_feat, t_feat)
    assert loss.item() >= 0


def test_feature_map_loss_with_adaptor():
    adaptor = FeatureAdaptor(student_channels=256, teacher_channels=512)
    t_feat = torch.randn(2, 512, 20, 20)
    s_feat = torch.randn(2, 256, 20, 20)
    loss = feature_map_loss(s_feat, t_feat, adaptor=adaptor)
    assert loss.item() >= 0


def test_feature_map_loss_spatial_mismatch():
    """Student can have different spatial size — should auto-interpolate."""
    t_feat = torch.randn(2, 512, 20, 20)
    s_feat = torch.randn(2, 512, 40, 40)   # 2× student spatial
    loss = feature_map_loss(s_feat, t_feat)
    assert loss.item() >= 0


def test_knowledge_distiller_combined_loss():
    teacher_channels = [512, 512]
    student_channels = [256, 256]
    distiller = KnowledgeDistiller(teacher_channels, student_channels)

    t_feats = [torch.randn(2, 512, 20, 20), torch.randn(2, 512, 10, 10)]
    s_feats = [torch.randn(2, 256, 20, 20), torch.randn(2, 256, 10, 10)]
    t_logits = torch.randn(2, 4)
    s_logits = torch.randn(2, 4)

    losses = distiller.distillation_loss(t_feats, s_feats, t_logits, s_logits)
    assert "total" in losses
    assert "feat_loss" in losses
    assert "soft_loss" in losses
    assert losses["total"].item() >= 0


def test_distiller_no_logits():
    """Should work without logits — soft_loss should be zero."""
    distiller = KnowledgeDistiller([512], [256])
    t_feats = [torch.randn(2, 512, 20, 20)]
    s_feats = [torch.randn(2, 256, 20, 20)]
    losses = distiller.distillation_loss(t_feats, s_feats)
    assert losses["soft_loss"].item() == 0.0


# ---------------------------------------------------------------------------
# Quantization tests
# ---------------------------------------------------------------------------

from distill.quantization import (
    benchmark_latency,
    count_parameters,
    PTQQuantizer,
)


def test_count_parameters():
    model = ToyStudent()
    n = count_parameters(model)
    assert n > 0


def test_benchmark_latency():
    model = ToyStudent().eval()
    result = benchmark_latency(model, input_shape=(1, 3, 32, 32),
                                n_warmup=2, n_runs=5, device="cpu")
    assert "mean_ms" in result
    assert result["mean_ms"] > 0
    assert result["fps"] > 0


def test_ptq_requires_calibration():
    model = ToyStudent()
    ptq = PTQQuantizer(model)
    with pytest.raises(RuntimeError, match="calibrate"):
        ptq.quantize()


# ---------------------------------------------------------------------------
# Benchmark runner tests
# ---------------------------------------------------------------------------

from distill.benchmark import BenchmarkRunner, BenchmarkResult


def test_benchmark_result_dataclass():
    r = BenchmarkResult(name="test", map50=0.9, fps=45.0)
    assert r.name == "test"
    assert r.map50 == 0.9


def test_benchmark_runner_latency_only():
    """Runner should work without val_loader (mAP returns -1)."""
    runner = BenchmarkRunner(
        val_loader=None,
        class_names=["a", "b"],
        device="cpu",
        n_warmup=1,
        n_latency_runs=3,
    )
    model = ToyStudent()
    runner.add("student", model)
    results = runner.run()
    assert len(results) == 1
    assert results[0].mean_latency_ms > 0


def test_benchmark_print_table(capsys):
    results = [
        BenchmarkResult("teacher", map50=0.981, fps=15.0, model_size_mb=22.0),
        BenchmarkResult("student", map50=0.965, fps=42.0, model_size_mb=6.0,
                        compression_ratio=3.5),
    ]
    BenchmarkRunner.print_table(results)
    captured = capsys.readouterr()
    assert "teacher" in captured.out
    assert "student" in captured.out


def test_benchmark_save_csv(tmp_path):
    results = [BenchmarkResult("test", map50=0.9)]
    out = tmp_path / "results.csv"
    BenchmarkRunner.save_csv(results, out)
    assert out.exists()
    content = out.read_text()
    assert "test" in content
