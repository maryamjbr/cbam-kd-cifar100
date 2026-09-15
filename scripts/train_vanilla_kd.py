#!/usr/bin/env python3
"""Train ResNet-20 with vanilla, logit-only knowledge distillation."""

from __future__ import annotations

import argparse

import torch
from torch import nn

from cbam_kd.checkpoint import load_model_checkpoint, save_training_checkpoint
from cbam_kd.config import load_config, section
from cbam_kd.data import build_loaders
from cbam_kd.distillation import VanillaDistillationLoss
from cbam_kd.engine import evaluate, train_vanilla_kd_epoch
from cbam_kd.models.resnet_cifar import build_resnet
from cbam_kd.utils import append_csv, ensure_directory, resolve_device, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--student-initial-checkpoint", default=None)
    parser.add_argument("--device", default=None)
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
    data_cfg = section(config, "data")
    teacher_cfg = section(config, "teacher")
    student_cfg = section(config, "student")
    distill_cfg = section(config, "distillation")
    train_cfg = section(config, "training")

    if bool(distill_cfg.get("cbam", False)):
        raise ValueError("Vanilla KD requires distillation.cbam=false")
    if bool(distill_cfg.get("feature_loss", False)):
        raise ValueError("Vanilla KD requires distillation.feature_loss=false")

    seed_everything(
        int(experiment.get("seed", 42)),
        deterministic=bool(experiment.get("deterministic", False)),
    )
    device = resolve_device(args.device or str(experiment.get("device", "auto")))
    output_dir = ensure_directory(experiment.get("output_dir", "runs/vanilla_kd"))
    loaders = build_loaders(data_cfg)

    teacher = build_resnet(
        str(teacher_cfg.get("name", "resnet56")),
        num_classes=int(teacher_cfg.get("num_classes", 100)),
    ).to(device)
    teacher_checkpoint = args.teacher_checkpoint or teacher_cfg.get("checkpoint")
    if not teacher_checkpoint:
        raise ValueError("A teacher checkpoint is required")
    load_model_checkpoint(teacher, teacher_checkpoint, map_location=device, strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student = build_resnet(
        str(student_cfg.get("name", "resnet20")),
        num_classes=int(student_cfg.get("num_classes", 100)),
    ).to(device)
    initial = args.student_initial_checkpoint or student_cfg.get("initial_checkpoint")
    if initial:
        load_model_checkpoint(student, initial, map_location=device, strict=True)

    criterion = VanillaDistillationLoss(
        temperature=float(distill_cfg.get("temperature", 4.0)),
        ce_weight=float(distill_cfg.get("ce_weight", 0.5)),
        kd_weight=float(distill_cfg.get("kd_weight", 0.5)),
        label_smoothing=float(distill_cfg.get("label_smoothing", 0.1)),
    ).to(device)
    optimizer = torch.optim.SGD(
        student.parameters(),
        lr=float(train_cfg.get("learning_rate", 0.1)),
        momentum=float(train_cfg.get("momentum", 0.9)),
        weight_decay=float(train_cfg.get("weight_decay", 5e-4)),
        nesterov=bool(train_cfg.get("nesterov", False)),
    )
    scheduler = build_scheduler(optimizer, train_cfg)
    evaluation_criterion = nn.CrossEntropyLoss()

    epochs = int(train_cfg.get("epochs", 250))
    patience = int(train_cfg.get("patience", 0))
    best_top1 = -1.0
    stale = 0
    for epoch in range(1, epochs + 1):
        train_metrics = train_vanilla_kd_epoch(
            student,
            teacher,
            loaders.train,
            optimizer,
            criterion,
            device,
            grad_clip=train_cfg.get("grad_clip"),
        )
        validation_metrics = evaluate(
            student,
            loaders.validation,
            device,
            evaluation_criterion,
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in validation_metrics.items()},
        }
        append_csv(output_dir / "history.csv", row)
        print(row, flush=True)

        if validation_metrics["top1"] > best_top1:
            best_top1 = validation_metrics["top1"]
            stale = 0
            save_training_checkpoint(
                output_dir / "best.pth",
                model=student,
                epoch=epoch,
                metrics=validation_metrics,
                config=config,
                optimizer=optimizer,
                scheduler=scheduler,
            )
        else:
            stale += 1
        if patience > 0 and stale >= patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    load_model_checkpoint(student, output_dir / "best.pth", map_location=device)
    test_metrics = evaluate(student, loaders.test, device, evaluation_criterion)
    write_json(output_dir / "test_metrics.json", test_metrics)
    print({"test": test_metrics}, flush=True)


if __name__ == "__main__":
    main()
