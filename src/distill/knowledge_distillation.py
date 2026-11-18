"""Knowledge Distillation for YOLOv8 — teacher → student feature-map KD.

Implements two complementary losses:

* **Soft-label KD** (Hinton et al., 2015): student learns from teacher's
  softened output logits, not just hard ground-truth labels.
* **Feature-map KD** (FitNets / PKD style): intermediate feature-map
  alignment between teacher and student at matched backbone stages.

The combination consistently outperforms soft-label-only distillation on
detection tasks, because it transfers spatial knowledge, not just class
probabilities.

Usage
-----
    teacher = YOLO("yolov8s.pt")
    student = YOLO("yolov8n.pt")
    kd = KnowledgeDistiller(teacher.model, student.model, temperature=4.0)
    loss = kd.distillation_loss(teacher_feats, student_feats,
                                teacher_logits, student_logits)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


__all__ = [
    "KnowledgeDistiller",
    "FeatureAdaptor",
    "soft_label_loss",
    "feature_map_loss",
]


class FeatureAdaptor(nn.Module):
    """1×1 conv to match student channel dim to teacher's.

    When teacher=YOLOv8s and student=YOLOv8n the backbone channel widths
    differ (e.g. 512 vs 256 at P4). A simple projection makes the L2
    comparison meaningful without distorting the student's activations.
    """

    def __init__(self, student_channels: int, teacher_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(student_channels, teacher_channels,
                              kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj.weight, mode="fan_out")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def soft_label_loss(student_logits: torch.Tensor,
                    teacher_logits: torch.Tensor,
                    temperature: float = 4.0) -> torch.Tensor:
    """KL-divergence KD loss on softened classification logits.

    Args:
        student_logits: raw cls logits from student head, shape (N, C).
        teacher_logits: raw cls logits from teacher head, shape (N, C).
        temperature: T. Higher values soften the distribution more.

    Returns:
        Scalar KD loss (mean over batch).
    """
    T = temperature
    p_teacher = F.softmax(teacher_logits / T, dim=-1)
    log_p_student = F.log_softmax(student_logits / T, dim=-1)
    # Multiply by T² to preserve gradient magnitudes (Hinton et al.)
    return F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T * T)


def feature_map_loss(student_feat: torch.Tensor,
                     teacher_feat: torch.Tensor,
                     adaptor: Optional[FeatureAdaptor] = None,
                     normalise: bool = True) -> torch.Tensor:
    """MSE loss between teacher and student feature maps at one stage.

    Args:
        student_feat: student intermediate feature, (B, Cs, H, W).
        teacher_feat: teacher intermediate feature, (B, Ct, H, W).
        adaptor: optional FeatureAdaptor to project Cs → Ct.
        normalise: L2-normalise across channel dim before MSE (improves
            stability when teacher/student scale differently).

    Returns:
        Scalar MSE loss.
    """
    if adaptor is not None:
        student_feat = adaptor(student_feat)

    # Spatial size must match — downsample student if needed
    if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
        student_feat = F.interpolate(student_feat,
                                     size=teacher_feat.shape[-2:],
                                     mode="bilinear",
                                     align_corners=False)

    if normalise:
        student_feat = F.normalize(student_feat, dim=1)
        teacher_feat = F.normalize(teacher_feat, dim=1)

    return F.mse_loss(student_feat, teacher_feat)


class KnowledgeDistiller(nn.Module):
    """Combined soft-label + feature-map KD manager.

    Keeps adaptors as a ModuleList so they are registered as parameters
    and move correctly with `.to(device)`.

    Args:
        teacher_channels: list of teacher channel widths at hook points,
            e.g. [256, 512, 1024] for P3/P4/P5.
        student_channels: matching list for student.
        temperature: softmax temperature for soft-label KD.
        alpha: weight of soft-label KD loss.
        beta: weight of feature-map KD loss.
    """

    def __init__(
        self,
        teacher_channels: List[int],
        student_channels: List[int],
        temperature: float = 4.0,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        super().__init__()
        assert len(teacher_channels) == len(student_channels), (
            "Must supply one channel width per hook point for both models."
        )
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

        self.adaptors = nn.ModuleList([
            FeatureAdaptor(sc, tc)
            for sc, tc in zip(student_channels, teacher_channels)
        ])

    def distillation_loss(
        self,
        teacher_feats: List[torch.Tensor],
        student_feats: List[torch.Tensor],
        teacher_logits: Optional[torch.Tensor] = None,
        student_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined KD loss.

        Returns a dict with keys ``feat_loss``, ``soft_loss``, ``total``
        so callers can log each component separately.
        """
        # Feature-map loss across all hook points
        feat_losses = [
            feature_map_loss(sf, tf, adaptor=adap)
            for sf, tf, adap in zip(student_feats, teacher_feats, self.adaptors)
        ]
        feat_loss = torch.stack(feat_losses).mean()

        # Soft-label loss (optional — skip if logits not provided)
        if teacher_logits is not None and student_logits is not None:
            soft_loss = soft_label_loss(student_logits, teacher_logits,
                                        temperature=self.temperature)
        else:
            soft_loss = torch.zeros(1, device=feat_loss.device)

        total = self.beta * feat_loss + self.alpha * soft_loss
        return {"feat_loss": feat_loss, "soft_loss": soft_loss, "total": total}
