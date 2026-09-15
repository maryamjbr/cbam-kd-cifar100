"""Training and evaluation loops."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from .distillation import DistillationLoss, VanillaDistillationLoss
from .metrics import ClassificationMeter


def _move_batch(batch: tuple[Tensor, Tensor], device: torch.device) -> tuple[Tensor, Tensor]:
    images, labels = batch
    non_blocking = device.type == "cuda"
    return (
        images.to(device, non_blocking=non_blocking),
        labels.to(device, non_blocking=non_blocking),
    )


def train_baseline_epoch(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float | None = None,
) -> dict[str, float]:
    model.train()
    meter = ClassificationMeter()
    progress = tqdm(loader, desc="train", leave=False)
    for batch in progress:
        images, labels = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        meter.update(logits.detach(), labels, float(loss.detach().item()), images.shape[0])
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return meter.compute()


def train_kd_epoch(
    student: Any,
    teacher: Any,
    loader: Iterable[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: DistillationLoss,
    device: torch.device,
    grad_clip: float | None = None,
) -> dict[str, float]:
    student.train()
    teacher.eval()
    criterion.train()
    meter = ClassificationMeter()
    sums = {"ce_loss": 0.0, "kd_loss": 0.0, "feature_loss": 0.0}

    progress = tqdm(loader, desc="train-kd", leave=False)
    for batch in progress:
        images, labels = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher_logits, teacher_features = teacher.forward_with_features(images)
        student_logits, student_features = student.forward_with_features(images)
        losses = criterion(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            student_features=student_features,
            teacher_features=teacher_features,
        )
        losses.total.backward()
        if grad_clip is not None:
            parameters = list(student.parameters()) + list(criterion.feature_distiller.parameters())
            torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
        optimizer.step()

        batch_size = images.shape[0]
        details = losses.detached()
        meter.update(student_logits.detach(), labels, details["loss"], batch_size)
        for key in sums:
            sums[key] += details[key] * batch_size
        progress.set_postfix(loss=f"{details['loss']:.4f}")

    metrics = meter.compute()
    if meter.samples:
        metrics.update({key: value / meter.samples for key, value in sums.items()})
    return metrics


def train_vanilla_kd_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: VanillaDistillationLoss,
    device: torch.device,
    grad_clip: float | None = None,
) -> dict[str, float]:
    """Train one epoch using logits only; no feature maps or CBAM are evaluated."""
    student.train()
    teacher.eval()
    criterion.train()
    meter = ClassificationMeter()
    sums = {"ce_loss": 0.0, "kd_loss": 0.0, "feature_loss": 0.0}

    progress = tqdm(loader, desc="train-vanilla-kd", leave=False)
    for batch in progress:
        images, labels = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher_logits = teacher(images)
        student_logits = student(images)
        losses = criterion(student_logits, teacher_logits, labels)
        losses.total.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
        optimizer.step()

        batch_size = images.shape[0]
        details = losses.detached()
        meter.update(student_logits.detach(), labels, details["loss"], batch_size)
        for key in sums:
            sums[key] += details[key] * batch_size
        progress.set_postfix(loss=f"{details['loss']:.4f}")

    metrics = meter.compute()
    if meter.samples:
        metrics.update({key: value / meter.samples for key, value in sums.items()})
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    model.eval()
    meter = ClassificationMeter()
    progress = tqdm(loader, desc="evaluate", leave=False)
    for batch in progress:
        images, labels = _move_batch(batch, device)
        logits = model(images)
        loss = 0.0 if criterion is None else float(criterion(logits, labels).item())
        meter.update(logits, labels, loss, images.shape[0])
    return meter.compute()
