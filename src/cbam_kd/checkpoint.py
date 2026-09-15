"""Checkpoint loading/saving compatible with both bare state_dicts and bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def _torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_model_state(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model_state_dict", "student_state_dict", "state_dict"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                return candidate
        if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
            return payload
    raise ValueError("Checkpoint does not contain a recognizable model state_dict")


def load_model_checkpoint(
    model: nn.Module,
    path: str | Path,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> Any:
    payload = _torch_load(path, map_location=map_location)
    state = extract_model_state(payload)
    model.load_state_dict(state, strict=strict)
    return payload


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    distiller: nn.Module | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if distiller is not None:
        payload["distiller_state_dict"] = distiller.state_dict()
    torch.save(payload, output)
