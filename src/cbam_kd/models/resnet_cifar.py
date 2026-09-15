"""CIFAR-style ResNet models compatible with the thesis checkpoints.

Attribution
-----------
The CIFAR stage layout (16 -> 32 -> 64 channels and [3, 3, 3]/[9, 9, 9]
blocks) follows Yerlan Idelbayev's CIFAR ResNet implementation:
https://github.com/akamaster/pytorch_resnet_cifar10

The ``downsample``-based BasicBlock structure, adaptive average pooling and
weight initialization follow the style of TorchVision's ResNet implementation:
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

Local modifications include CIFAR-100 output heads, projection shortcuts
(Option B), and explicit feature-return methods for knowledge distillation.
See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import torch
from torch import Tensor, nn

_STAGE_CHANNELS: Final[tuple[int, int, int]] = (16, 32, 64)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """Return a 3x3 convolution with one-pixel padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    """Two-convolution residual block used by CIFAR ResNets."""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class CifarResNet(nn.Module):
    """ResNet for 32x32 CIFAR images using projection shortcuts (Option B)."""

    def __init__(
        self,
        layers: Sequence[int],
        num_classes: int = 100,
    ) -> None:
        super().__init__()
        if len(layers) != 3:
            raise ValueError(f"Expected three stage depths, got {list(layers)}")

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(16, int(layers[0]), stride=1)
        self.layer2 = self._make_layer(32, int(layers[1]), stride=2)
        self.layer3 = self._make_layer(64, int(layers[2]), stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Sequential is retained for exact compatibility with the uploaded
        # checkpoints, whose classifier keys are ``fc.0.weight`` and ``fc.0.bias``.
        self.fc = nn.Sequential(nn.Linear(64, num_classes))

        self._initialize_weights()

    @property
    def feature_channels(self) -> tuple[int, int, int]:
        return _STAGE_CHANNELS

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample: nn.Module | None = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )

        layers: list[nn.Module] = [
            BasicBlock(self.inplanes, planes, stride=stride, downsample=downsample)
        ]
        self.inplanes = planes
        layers.extend(BasicBlock(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward_features(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return the outputs of the three residual stages."""
        x = self.relu(self.bn1(self.conv1(x)))
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        return feat1, feat2, feat3

    def classify_features(self, final_feature: Tensor) -> Tensor:
        x = self.avgpool(final_feature)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def forward_with_features(
        self,
        x: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        features = self.forward_features(x)
        logits = self.classify_features(features[-1])
        return logits, features

    def forward(self, x: Tensor) -> Tensor:
        logits, _ = self.forward_with_features(x)
        return logits


def resnet20(num_classes: int = 100) -> CifarResNet:
    return CifarResNet([3, 3, 3], num_classes=num_classes)


def resnet56(num_classes: int = 100) -> CifarResNet:
    return CifarResNet([9, 9, 9], num_classes=num_classes)


def resnet110(num_classes: int = 100) -> CifarResNet:
    return CifarResNet([18, 18, 18], num_classes=num_classes)


def build_resnet(name: str, num_classes: int = 100) -> CifarResNet:
    builders = {
        "resnet20": resnet20,
        "resnet56": resnet56,
        "resnet110": resnet110,
    }
    try:
        return builders[name.lower()](num_classes=num_classes)
    except KeyError as exc:
        choices = ", ".join(sorted(builders))
        raise ValueError(f"Unknown model {name!r}. Choose one of: {choices}") from exc
