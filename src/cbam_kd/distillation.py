"""Losses for CBAM-guided feature and logit knowledge distillation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .models.cbam import CBAM


@dataclass(frozen=True)
class LossBreakdown:
    total: Tensor
    cross_entropy: Tensor
    logit_kd: Tensor
    feature: Tensor

    def detached(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach().item()),
            "ce_loss": float(self.cross_entropy.detach().item()),
            "kd_loss": float(self.logit_kd.detach().item()),
            "feature_loss": float(self.feature.detach().item()),
        }


class VanillaDistillationLoss(nn.Module):
    """Logit-only knowledge distillation with no feature or attention branch."""

    def __init__(
        self,
        temperature: float = 4.0,
        ce_weight: float = 0.5,
        kd_weight: float = 0.5,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if ce_weight < 0 or kd_weight < 0:
            raise ValueError("Loss weights must be non-negative")
        if ce_weight + kd_weight <= 0:
            raise ValueError("At least one loss weight must be positive")

        self.temperature = float(temperature)
        self.ce_weight = float(ce_weight)
        self.kd_weight = float(kd_weight)
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        labels: Tensor,
    ) -> LossBreakdown:
        if labels.ndim == 2:
            labels = labels.argmax(dim=1)

        ce_loss = self.cross_entropy(student_logits, labels)
        temperature = self.temperature
        teacher_probability = F.softmax(teacher_logits.detach() / temperature, dim=1)
        student_log_probability = F.log_softmax(student_logits / temperature, dim=1)
        kd_loss = F.kl_div(
            student_log_probability,
            teacher_probability,
            reduction="batchmean",
        ) * (temperature**2)
        feature_loss = student_logits.new_zeros(())
        total = self.ce_weight * ce_loss + self.kd_weight * kd_loss
        return LossBreakdown(total, ce_loss, kd_loss, feature_loss)


class CBAMFeatureDistiller(nn.Module):
    """Apply stage-specific CBAM modules only in the distillation branch.

    The student inference graph remains an ordinary ResNet. The same CBAM module
    is used for the teacher and student feature map at each stage. Teacher-side
    CBAM outputs are computed without gradient; CBAM parameters are optimized
    through the student branch and are therefore included in the KD optimizer.
    """

    def __init__(
        self,
        channels: Sequence[int] = (16, 32, 64),
        reduction: int = 16,
        spatial_kernel: int = 7,
        distance: str = "mse",
        stage_weights: Sequence[float] = (1.0, 2.0, 3.0),
        sample_weighting: str = "none",
        sample_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if len(channels) == 0:
            raise ValueError("At least one feature stage is required")
        if len(stage_weights) != len(channels):
            raise ValueError("stage_weights must match the number of feature stages")
        if any(weight < 0 for weight in stage_weights):
            raise ValueError("stage_weights must be non-negative")
        if sum(stage_weights) <= 0:
            raise ValueError("At least one stage weight must be positive")
        if sample_temperature <= 0:
            raise ValueError("sample_temperature must be positive")

        distance = distance.lower()
        if distance not in {"mse", "l1", "smooth_l1"}:
            raise ValueError("distance must be one of: mse, l1, smooth_l1")
        sample_weighting = sample_weighting.lower()
        if sample_weighting not in {"none", "cosine", "legacy_dot"}:
            raise ValueError("sample_weighting must be none, cosine or legacy_dot")

        self.blocks = nn.ModuleList(
            [CBAM(ch, reduction=reduction, spatial_kernel=spatial_kernel) for ch in channels]
        )
        normalized = torch.tensor(stage_weights, dtype=torch.float32)
        normalized = normalized / normalized.sum()
        self.register_buffer("stage_weights", normalized)
        self.distance = distance
        self.sample_weighting = sample_weighting
        self.sample_temperature = float(sample_temperature)

    def _per_sample_distance(self, student: Tensor, teacher: Tensor) -> Tensor:
        reduce_dims = tuple(range(1, student.ndim))
        if self.distance == "mse":
            return (student - teacher).pow(2).mean(dim=reduce_dims)
        if self.distance == "l1":
            return (student - teacher).abs().mean(dim=reduce_dims)
        return F.smooth_l1_loss(student, teacher, reduction="none").mean(dim=reduce_dims)

    def _sample_weights(self, student: Tensor, teacher: Tensor) -> Tensor:
        if self.sample_weighting == "none":
            return torch.ones(student.shape[0], device=student.device, dtype=student.dtype)

        student_vec = student.mean(dim=(-2, -1))
        teacher_vec = teacher.mean(dim=(-2, -1))
        if self.sample_weighting == "cosine":
            similarity = F.cosine_similarity(student_vec, teacher_vec, dim=1)
            # Convert [-1, 1] to [0, 1], preserving a usable gradient scale.
            return ((similarity + 1.0) / 2.0).detach()

        logits = (student_vec * teacher_vec).sum(dim=1) / self.sample_temperature
        return torch.sigmoid(logits).detach()

    def forward(
        self,
        teacher_features: Sequence[Tensor],
        student_features: Sequence[Tensor],
    ) -> Tensor:
        if len(teacher_features) != len(self.blocks) or len(student_features) != len(self.blocks):
            raise ValueError(
                "Teacher/student feature counts must match the configured CBAM stages"
            )

        total = torch.zeros((), device=student_features[0].device)
        for index, (block, teacher_raw, student_raw) in enumerate(
            zip(self.blocks, teacher_features, student_features, strict=True)
        ):
            if teacher_raw.shape != student_raw.shape:
                raise ValueError(
                    f"Feature shape mismatch at stage {index}: "
                    f"teacher={tuple(teacher_raw.shape)}, student={tuple(student_raw.shape)}"
                )
            with torch.no_grad():
                teacher_attended = block(teacher_raw.detach())
            student_attended = block(student_raw)
            per_sample = self._per_sample_distance(student_attended, teacher_attended)
            sample_weights = self._sample_weights(student_attended, teacher_attended)
            stage_loss = (per_sample * sample_weights).mean()
            total = total + self.stage_weights[index] * stage_loss
        return total


class DistillationLoss(nn.Module):
    """Weighted combination of CE, temperature-scaled KL and CBAM feature loss."""

    def __init__(
        self,
        feature_distiller: CBAMFeatureDistiller,
        temperature: float = 4.0,
        ce_weight: float = 0.5,
        kd_weight: float = 0.2,
        feature_weight: float = 0.3,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        weights = (ce_weight, kd_weight, feature_weight)
        if any(value < 0 for value in weights):
            raise ValueError("Loss weights must be non-negative")
        if sum(weights) <= 0:
            raise ValueError("At least one loss weight must be positive")

        self.feature_distiller = feature_distiller
        self.temperature = float(temperature)
        self.ce_weight = float(ce_weight)
        self.kd_weight = float(kd_weight)
        self.feature_weight = float(feature_weight)
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        labels: Tensor,
        student_features: Sequence[Tensor],
        teacher_features: Sequence[Tensor],
    ) -> LossBreakdown:
        if labels.ndim == 2:
            labels = labels.argmax(dim=1)

        ce_loss = self.cross_entropy(student_logits, labels)
        temperature = self.temperature
        teacher_probability = F.softmax(teacher_logits.detach() / temperature, dim=1)
        student_log_probability = F.log_softmax(student_logits / temperature, dim=1)
        kd_loss = F.kl_div(
            student_log_probability,
            teacher_probability,
            reduction="batchmean",
        ) * (temperature**2)
        feature_loss = self.feature_distiller(teacher_features, student_features)
        total = (
            self.ce_weight * ce_loss
            + self.kd_weight * kd_loss
            + self.feature_weight * feature_loss
        )
        return LossBreakdown(total, ce_loss, kd_loss, feature_loss)
