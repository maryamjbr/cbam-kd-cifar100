from pathlib import Path

import pytest
import torch

from cbam_kd.checkpoint import load_model_checkpoint
from cbam_kd.distillation import CBAMFeatureDistiller, DistillationLoss, VanillaDistillationLoss
from cbam_kd.models.resnet_cifar import resnet20, resnet56


ROOT = Path(__file__).resolve().parents[1]


def test_model_shapes_and_parameter_counts():
    student = resnet20(100)
    teacher = resnet56(100)
    assert sum(parameter.numel() for parameter in student.parameters()) == 278_324
    assert sum(parameter.numel() for parameter in teacher.parameters()) == 861_620

    images = torch.randn(2, 3, 32, 32)
    logits, features = student.forward_with_features(images)
    assert logits.shape == (2, 100)
    assert [tuple(feature.shape) for feature in features] == [
        (2, 16, 32, 32),
        (2, 32, 16, 16),
        (2, 64, 8, 8),
    ]


CHECKPOINTS = [
    ("resnet20_cifar100_best.pth", resnet20),
    ("resnet20_kd_best.pth", resnet20),
    ("resnet56_cifar100_best.pth", resnet56),
]


@pytest.mark.skipif(
    not all((ROOT / "checkpoints" / name).exists() for name, _ in CHECKPOINTS),
    reason="Historical checkpoints are optional and are not stored in Git",
)
def test_optional_checkpoints_load_exactly():
    for name, builder in CHECKPOINTS:
        model = builder(100)
        load_model_checkpoint(model, ROOT / "checkpoints" / name, strict=True)


def test_kd_backward_reaches_student_and_cbam():
    student = resnet20(100)
    teacher = resnet56(100)
    distiller = CBAMFeatureDistiller(sample_weighting="cosine")
    criterion = DistillationLoss(distiller)

    images = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([1, 2])
    with torch.no_grad():
        teacher_logits, teacher_features = teacher.forward_with_features(images)
    student_logits, student_features = student.forward_with_features(images)
    losses = criterion(
        student_logits,
        teacher_logits,
        labels,
        student_features,
        teacher_features,
    )
    losses.total.backward()
    assert any(parameter.grad is not None for parameter in student.parameters())
    assert any(parameter.grad is not None for parameter in distiller.parameters())


def test_vanilla_kd_has_no_attention_or_feature_loss():
    student = resnet20(100)
    teacher = resnet56(100)
    criterion = VanillaDistillationLoss(
        temperature=4.0,
        ce_weight=0.5,
        kd_weight=0.5,
        label_smoothing=0.1,
    )

    images = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([1, 2])
    with torch.no_grad():
        teacher_logits = teacher(images)
    losses = criterion(student(images), teacher_logits, labels)
    losses.total.backward()

    assert losses.feature.item() == 0.0
    assert not any("cbam" in name.lower() for name, _ in criterion.named_modules())
    assert any(parameter.grad is not None for parameter in student.parameters())
