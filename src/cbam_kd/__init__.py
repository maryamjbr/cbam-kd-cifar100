"""CBAM-guided knowledge distillation for CIFAR-100."""

from .distillation import CBAMFeatureDistiller, DistillationLoss, VanillaDistillationLoss
from .models.resnet_cifar import build_resnet, resnet20, resnet56, resnet110

__all__ = [
    "CBAMFeatureDistiller",
    "DistillationLoss",
    "VanillaDistillationLoss",
    "build_resnet",
    "resnet20",
    "resnet56",
    "resnet110",
]

__version__ = "0.1.0"
