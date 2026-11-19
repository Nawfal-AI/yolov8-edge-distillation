"""Unit tests for the distillation and quantization modules.

All tests use minimal toy models — no real weights, no ultralytics,
no dataset required. Run with: pytest tests/ -v
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from distill.knowledge_distillation import (
    KnowledgeDistiller,
    FeatureAdaptor,
    soft_label_loss,
    feature_map_loss,
)
from distill.quantization import (
    benchmark_latency,
    count_parameters,
    PTQQuantizer,
)
from distill.benchmark import BenchmarkRunner, BenchmarkResult


# ---------------------------------------------------------------------------
# Minimal toy models — no real weights needed
# ---------------------------------------------------------------------------

class ToyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 512, 3, padding=1)
        self.bn   = nn.BatchNorm2d(512)
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
        self.bn   = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.head = nn.Linear(256, 4)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = x.mean(dim=(2, 3))
        return self.head(x)


# ---------------------------------------------------------------------------
# KD loss tests
# ---------------------------------------------------------------------------

def test_soft_label_loss_returns_scalar():
    logits_t = torch.randn(8, 4)
    logits_s = torch.randn(8, 4)
    loss = soft_label_loss(logits_t, logits_s, temperature=4.0)
    assert loss.shape == (), "must return a scalar"
    assert loss.item() >= 0,  "KL divergence must be non-negative"


def test_feature_map_loss_same_channels():
    t = torch.randn(2, 512, 20, 20)
    s = torch.randn(2, 512, 20, 20)
    assert feature_map_loss(s, t).item() >= 0


def test_feature_map_loss_with_adaptor():
    adaptor = FeatureAdaptor(student_channels=256, teacher_channels=512)
    t = torch.randn(2, 512, 20, 20)
    s = torch.randn(2, 256, 20, 20)
    assert feature_map_loss(s, t, adaptor=adaptor).item() >= 0


def test_feature_map_loss_spatial_mismatch():
    """Student with 2× spatial size should auto-interpolate."""
    t = torch.randn(2, 512, 20, 20)
    s = torch.randn(2, 512, 40, 40)
    assert feature_map_loss(s, t).item() >= 0


def test_knowledge_distiller_combined_loss():
    distiller = KnowledgeDistiller([512, 512], [256, 256])
    t_feats  = [torch.randn(2, 512, 20, 20), torch.randn(2, 512, 10, 10)]
    s_feats  = [torch.randn(2, 256, 20, 20), torch.randn(2, 256, 10, 10)]
    t_logits = torch.randn(2, 4)
    s_logits = torch.randn(2, 4)
    losses = distiller.distillation_loss(t_feats, s_feats, t_logits, s_logits)
    assert "total"     in losses
    assert "feat_loss" in losses
    assert "soft_loss" in losses
    assert losses["total"].item() >= 0


def test_distiller_no_logits_gives_zero_soft_loss():
    distiller = KnowledgeDistiller([512], [256])
    losses = distiller.distillation_loss(
        [torch.randn(2, 512, 20, 20)],
        [torch.randn(2, 256, 20, 20)],
    )
    assert losses["soft_loss"].item() == 0.0


# ---------------------------------------------------------------------------
# Quantization tests
# ---------------------------------------------------------------------------

def test_count_parameters():
    assert count_parameters(ToyStudent()) > 0


def test_benchmark_latency():
    result = benchmark_latency(
        ToyStudent().eval(),
        input_shape=(1, 3, 32, 32),
        n_warmup=2, n_runs=5, device="cpu",
    )
    assert result["mean_ms"] > 0
    assert result["fps"]     > 0


def test_ptq_requires_calibration_before_quantize():
    ptq = PTQQuantizer(ToyStudent())
    with pytest.raises(RuntimeError, match="calibrate"):
        ptq.quantize()


# ---------------------------------------------------------------------------
# Benchmark runner tests
# ---------------------------------------------------------------------------

def test_benchmark_result_dataclass():
    r = BenchmarkResult(name="test", map50=0.9, fps=45.0)
    assert r.name   == "test"
    assert r.map50  == 0.9


def test_benchmark_runner_latency_only():
    runner = BenchmarkRunner(
        val_loader=None,
        class_names=["a", "b"],
        device="cpu",
        n_warmup=1,
        n_latency_runs=3,
    )
    runner.add("student", ToyStudent())
    results = runner.run()
    assert len(results) == 1
    assert results[0].mean_latency_ms > 0


def test_benchmark_print_table(capsys):
    BenchmarkRunner.print_table([
        BenchmarkResult("teacher", map50=0.9727, fps=58.5,  model_size_mb=21.47),
        BenchmarkResult("student", map50=0.9615, fps=120.9, model_size_mb=6.2,
                        compression_ratio=3.46),
    ])
    out = capsys.readouterr().out
    assert "teacher" in out
    assert "student" in out


def test_benchmark_save_csv(tmp_path):
    out = tmp_path / "results.csv"
    BenchmarkRunner.save_csv([BenchmarkResult("test", map50=0.9)], out)
    assert out.exists()
    assert "test" in out.read_text()
