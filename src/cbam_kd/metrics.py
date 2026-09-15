"""Streaming classification metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class ClassificationMeter:
    samples: int = 0
    correct_top1: int = 0
    correct_top5: int = 0
    loss_sum: float = 0.0

    def update(self, logits: Tensor, labels: Tensor, loss: float, batch_size: int) -> None:
        topk = min(5, logits.shape[1])
        predictions = logits.topk(topk, dim=1).indices
        matches = predictions.eq(labels.view(-1, 1))
        self.samples += batch_size
        self.correct_top1 += int(matches[:, :1].sum().item())
        self.correct_top5 += int(matches.sum().item())
        self.loss_sum += float(loss) * batch_size

    def compute(self) -> dict[str, float]:
        if self.samples == 0:
            return {"loss": 0.0, "top1": 0.0, "top5": 0.0}
        return {
            "loss": self.loss_sum / self.samples,
            "top1": self.correct_top1 / self.samples,
            "top5": self.correct_top5 / self.samples,
        }
