"""Knowledge Distillation loss functions.

Two complementary KD signals:

1. **SoftLabelKDLoss** — transfers the teacher's output probability
   distribution (class confidence scores) to the student via KL-divergence.
   Temperature T controls how much the teacher's uncertainty is shared.

2. **FeatureMapKDLoss** — transfers intermediate backbone representations
   via MSE between teacher and student feature maps at matched spatial
   resolutions. This is the dominant signal for small-object detection tasks
   because it teaches the student *how to see*, not just *what to predict*.

Usage::

    from src.distillation.losses import FeatureMapKDLoss, SoftLabelKDLoss

    fm_loss = FeatureMapKDLoss(weight=1.0)
    sl_loss = SoftLabelKDLoss(temperature=4.0, weight=0.5)

    total_kd = fm_loss(student_features, teacher_features) \\
             + sl_loss(student_logits, teacher_logits)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SoftLabelKDLoss", "FeatureMapKDLoss", "CombinedKDLoss"]


class SoftLabelKDLoss(nn.Module):
    """KL-divergence between teacher and student output distributions.

    Args:
        temperature: Softmax temperature T. Higher T produces softer
            distributions, sharing more of the teacher's uncertainty with
            the student. T=1 reduces to standard cross-entropy. T=4 is a
            common starting point for detection tasks.
        weight: Loss weight for combining with task loss.
    """

    def __init__(self, temperature: float = 4.0, weight: float = 0.5) -> None:
        super().__init__()
        self.T = temperature
        self.weight = weight

    def forward(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            student_logits: Raw (pre-softmax) class scores from student.
                Shape: (N, num_classes) or (N, A, num_classes).
            teacher_logits: Matching logits from teacher (no grad needed).

        Returns:
            Scalar KD loss weighted by self.weight * T^2.
        """
        # T^2 scaling preserves gradient magnitudes as T increases
        # (Hinton et al., 2015, Eq. 2)
        scale = self.T ** 2

        p_teacher = F.softmax(teacher_logits / self.T, dim=-1)
        log_p_student = F.log_softmax(student_logits / self.T, dim=-1)

        kd_loss = F.kl_div(log_p_student, p_teacher, reduction="batchmean")
        return self.weight * scale * kd_loss


class FeatureMapKDLoss(nn.Module):
    """MSE between teacher and student intermediate feature maps.

    For YOLOv8, we extract features at three scales from the backbone neck
    (P3, P4, P5 — small, medium, large object heads). The student maps are
    projected to the teacher's channel dimension via a 1x1 conv adapter
    before MSE, so the two architectures don't need matching channel counts.

    Args:
        student_channels: List of channel counts at each student feature level.
        teacher_channels: List of channel counts at each teacher feature level.
        weight: Loss weight for combining with task loss.
    """

    def __init__(
        self,
        student_channels: list[int] | None = None,
        teacher_channels: list[int] | None = None,
        weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight = weight

        # YOLOv8n -> YOLOv8s default channel dims at P3/P4/P5
        s_ch = student_channels or [64, 128, 256]
        t_ch = teacher_channels or [128, 256, 512]

        # 1x1 adapters project student channels up to teacher channels
        self.adapters = nn.ModuleList(
            [nn.Conv2d(s, t, kernel_size=1, bias=False) for s, t in zip(s_ch, t_ch)]
        )

    def forward(
        self,
        student_features: list[torch.Tensor],
        teacher_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            student_features: Feature maps from student neck [P3, P4, P5].
            teacher_features: Feature maps from teacher neck [P3, P4, P5].
                Teacher features should have no_grad() applied by the caller.

        Returns:
            Scalar MSE loss averaged across feature levels, weighted by self.weight.
        """
        assert len(student_features) == len(teacher_features) == len(self.adapters), (
            f"Expected 3 feature levels, got student={len(student_features)}, "
            f"teacher={len(teacher_features)}"
        )

        total = torch.tensor(0.0, device=student_features[0].device)
        for adapter, s_feat, t_feat in zip(
            self.adapters, student_features, teacher_features
        ):
            # Project student up to teacher channel dim
            s_proj = adapter(s_feat)

            # Spatial dims must match — teacher and student share input size
            if s_proj.shape[-2:] != t_feat.shape[-2:]:
                s_proj = F.interpolate(
                    s_proj, size=t_feat.shape[-2:], mode="bilinear", align_corners=False
                )

            total = total + F.mse_loss(s_proj, t_feat.detach())

        return self.weight * (total / len(self.adapters))


class CombinedKDLoss(nn.Module):
    """Convenience wrapper combining FeatureMapKDLoss and SoftLabelKDLoss.

    Total loss = task_loss + fm_kd_loss + sl_kd_loss

    The task_loss (standard YOLOv8 detection loss) is passed in from the
    trainer and added here so the combined scalar can be passed to .backward()
    in one call.

    Args:
        fm_weight: Weight for feature-map KD term.
        sl_weight: Weight for soft-label KD term.
        temperature: Softmax temperature for soft-label KD.
        student_channels: Student neck channel dims [P3, P4, P5].
        teacher_channels: Teacher neck channel dims [P3, P4, P5].
    """

    def __init__(
        self,
        fm_weight: float = 1.0,
        sl_weight: float = 0.5,
        temperature: float = 4.0,
        student_channels: list[int] | None = None,
        teacher_channels: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.fm_loss = FeatureMapKDLoss(
            student_channels=student_channels,
            teacher_channels=teacher_channels,
            weight=fm_weight,
        )
        self.sl_loss = SoftLabelKDLoss(temperature=temperature, weight=sl_weight)

    def forward(
        self,
        task_loss: torch.Tensor,
        student_features: list[torch.Tensor],
        teacher_features: list[torch.Tensor],
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            Dict with keys: total, task, fm_kd, sl_kd — for logging.
        """
        fm = self.fm_loss(student_features, teacher_features)
        sl = self.sl_loss(student_logits, teacher_logits)
        total = task_loss + fm + sl

        return {
            "total": total,
            "task": task_loss.detach(),
            "fm_kd": fm.detach(),
            "sl_kd": sl.detach(),
        }
