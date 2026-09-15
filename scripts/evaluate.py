#!/usr/bin/env python3
"""Evaluate a bare or bundled checkpoint on the untouched CIFAR-100 test set."""

from __future__ import annotations

import argparse

from torch import nn

from cbam_kd.checkpoint import load_model_checkpoint
from cbam_kd.config import load_config, section
from cbam_kd.data import build_loaders
from cbam_kd.engine import evaluate
from cbam_kd.models.resnet_cifar import build_resnet
from cbam_kd.utils import resolve_device, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = section(config, "data")
    model_cfg = section(config, "model")
    if not model_cfg:
        model_cfg = section(config, "student")

    device = resolve_device(args.device)
    model = build_resnet(
        args.model or str(model_cfg.get("name", "resnet20")),
        num_classes=int(model_cfg.get("num_classes", 100)),
    ).to(device)
    load_model_checkpoint(model, args.checkpoint, map_location=device, strict=True)
    loaders = build_loaders(data_cfg)
    metrics = evaluate(model, loaders.test, device, nn.CrossEntropyLoss())
    print(metrics)
    if args.output:
        write_json(args.output, metrics)


if __name__ == "__main__":
    main()
