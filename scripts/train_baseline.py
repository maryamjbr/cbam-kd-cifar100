#!/usr/bin/env python3
"""Train a CIFAR-100 ResNet teacher or student baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from cbam_kd.checkpoint import save_training_checkpoint
from cbam_kd.config import load_config, section
from cbam_kd.data import build_loaders
from cbam_kd.engine import evaluate, train_baseline_epoch
from cbam_kd.models.resnet_cifar import build_resnet
from cbam_kd.utils import append_csv, ensure_directory, resolve_device, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None, help="Override config device")
    return parser.parse_args()


def build_scheduler(optimizer: torch.optim.Optimizer, train_cfg: dict):
    name = str(train_cfg.get("scheduler", "multistep")).lower()
    if name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(x) for x in train_cfg.get("milestones", [150, 200])],
            gamma=float(train_cfg.get("gamma", 0.1)),
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(train_cfg.get("epochs", 250)),
            eta_min=float(train_cfg.get("eta_min", 1e-5)),
        )
    raise ValueError("scheduler must be multistep or cosine")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    experiment = section(config, "experiment")
    model_cfg = section(config, "model")
    data_cfg = section(config, "data")
    train_cfg = section(config, "training")

    seed = int(experiment.get("seed", 42))
    seed_everything(seed, deterministic=bool(experiment.get("deterministic", False)))
    device = resolve_device(args.device or str(experiment.get("device", "auto")))
    output_dir = ensure_directory(experiment.get("output_dir", "runs/baseline"))

    loaders = build_loaders(data_cfg)
    model = build_resnet(
        str(model_cfg.get("name", "resnet20")),
        num_classes=int(model_cfg.get("num_classes", 100)),
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0))
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 0.1)),
        momentum=float(train_cfg.get("momentum", 0.9)),
        weight_decay=float(train_cfg.get("weight_decay", 5e-4)),
        nesterov=bool(train_cfg.get("nesterov", False)),
    )
    scheduler = build_scheduler(optimizer, train_cfg)

    best_top1 = -1.0
    patience = int(train_cfg.get("patience", 0))
    stale = 0
    epochs = int(train_cfg.get("epochs", 250))
    for epoch in range(1, epochs + 1):
        train_metrics = train_baseline_epoch(
            model,
            loaders.train,
            optimizer,
            criterion,
            device,
            grad_clip=train_cfg.get("grad_clip"),
        )
        validation_metrics = evaluate(model, loaders.validation, device, criterion)
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in validation_metrics.items()},
        }
        append_csv(output_dir / "history.csv", row)
        print(row)

        if validation_metrics["top1"] > best_top1:
            best_top1 = validation_metrics["top1"]
            stale = 0
            save_training_checkpoint(
                output_dir / "best.pth",
                model=model,
                epoch=epoch,
                metrics=validation_metrics,
                config=config,
                optimizer=optimizer,
                scheduler=scheduler,
            )
        else:
            stale += 1
        if patience > 0 and stale >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    from cbam_kd.checkpoint import load_model_checkpoint

    load_model_checkpoint(model, output_dir / "best.pth", map_location=device)
    test_metrics = evaluate(model, loaders.test, device, criterion)
    write_json(output_dir / "test_metrics.json", test_metrics)
    print({"test": test_metrics})


if __name__ == "__main__":
    main()
