"""CIFAR-100 datasets, deterministic train/validation split and loaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


@dataclass(frozen=True)
class LoaderBundle:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


def build_transforms(use_randaugment: bool = False) -> tuple[transforms.Compose, transforms.Compose]:
    train_steps: list[Any] = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if use_randaugment:
        train_steps.append(transforms.RandAugment(num_ops=2, magnitude=10))
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    return transforms.Compose(train_steps), evaluation_transform


def build_loaders(config: dict[str, Any]) -> LoaderBundle:
    root = Path(config.get("root", "./data"))
    batch_size = int(config.get("batch_size", 128))
    workers = int(config.get("num_workers", 4))
    validation_size = int(config.get("validation_size", 5000))
    split_seed = int(config.get("split_seed", 42))
    use_randaugment = bool(config.get("randaugment", False))
    download = bool(config.get("download", True))
    pin_memory = bool(config.get("pin_memory", torch.cuda.is_available()))

    if not 0 < validation_size < 50_000:
        raise ValueError("validation_size must be between 1 and 49,999")

    train_transform, evaluation_transform = build_transforms(use_randaugment)
    train_source = datasets.CIFAR100(
        root=root,
        train=True,
        transform=train_transform,
        download=download,
    )
    validation_source = datasets.CIFAR100(
        root=root,
        train=True,
        transform=evaluation_transform,
        download=download,
    )
    test_source = datasets.CIFAR100(
        root=root,
        train=False,
        transform=evaluation_transform,
        download=download,
    )

    generator = torch.Generator().manual_seed(split_seed)
    permutation = torch.randperm(len(train_source), generator=generator).tolist()
    validation_indices = permutation[:validation_size]
    train_indices = permutation[validation_size:]

    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        Subset(train_source, train_indices),
        shuffle=True,
        drop_last=False,
        **common,
    )
    validation_loader = DataLoader(
        Subset(validation_source, validation_indices),
        shuffle=False,
        drop_last=False,
        **common,
    )
    test_loader = DataLoader(
        test_source,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return LoaderBundle(train_loader, validation_loader, test_loader)
