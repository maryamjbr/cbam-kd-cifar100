#!/usr/bin/env python3
"""Inspect architecture compatibility, tensor count and SHA-256 of checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from cbam_kd.checkpoint import extract_model_state
from cbam_kd.models.resnet_cifar import build_resnet


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    records = []
    for raw_path in args.paths:
        path = Path(raw_path)
        payload = torch_load(path)
        state = extract_model_state(payload)
        classifier = state.get("fc.0.weight")
        if classifier is None:
            raise ValueError(f"Cannot infer class count from {path}")
        num_classes = classifier.shape[0]
        architecture = "resnet56" if any(key.startswith("layer1.8.") for key in state) else "resnet20"
        model = build_resnet(architecture, num_classes=num_classes)
        incompatible = model.load_state_dict(state, strict=False)
        records.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "architecture": architecture,
                "num_classes": int(num_classes),
                "tensor_elements": int(sum(value.numel() for value in state.values())),
                "missing_keys": incompatible.missing_keys,
                "unexpected_keys": incompatible.unexpected_keys,
            }
        )

    print(json.dumps(records, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
