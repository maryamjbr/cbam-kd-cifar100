#!/usr/bin/env python3
"""Run vanilla KD from matched checkpoints, then evaluate, plot, and archive results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def run_logged(command: list[str], log_path: Path, *, cwd: Path) -> None:
    print(f"\n$ {' '.join(command)}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_checkpoint(reference_root: Path, seed: int, model_dir: str) -> Path:
    suffix = Path(f"seed_{seed}/{model_dir}/best.pth")
    matches = sorted(path for path in reference_root.glob("**/best.pth") if str(path).endswith(str(suffix)))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {model_dir} checkpoint ending in {suffix}, found {matches}"
        )
    return matches[0]


def configure(base: dict[str, Any], seed: int, output_dir: Path, data_root: Path, workers: int):
    config = json.loads(json.dumps(base))
    config["experiment"]["seed"] = seed
    config["experiment"]["output_dir"] = str(output_dir)
    config["data"]["root"] = str(data_root)
    config["data"]["num_workers"] = workers
    config["data"]["download"] = True
    return config


def validate_contract(config: dict[str, Any]) -> dict[str, Any]:
    contract = {
        "teacher": config["teacher"]["name"],
        "student": config["student"]["name"],
        "dataset": "CIFAR-100",
        "validation_size": config["data"]["validation_size"],
        "split_seed": config["data"]["split_seed"],
        "epochs": config["training"]["epochs"],
        "batch_size": config["data"]["batch_size"],
        "optimizer": "SGD",
        "learning_rate": config["training"]["learning_rate"],
        "momentum": config["training"]["momentum"],
        "weight_decay": config["training"]["weight_decay"],
        "lr_decay_epochs": config["training"]["milestones"],
        "temperature": config["distillation"]["temperature"],
        "label_smoothing": config["distillation"]["label_smoothing"],
        "ce_weight": config["distillation"]["ce_weight"],
        "kd_weight": config["distillation"]["kd_weight"],
        "cbam": config["distillation"]["cbam"],
        "feature_loss": config["distillation"]["feature_loss"],
    }
    expected = {
        "teacher": "resnet56",
        "student": "resnet20",
        "dataset": "CIFAR-100",
        "validation_size": 5000,
        "split_seed": 42,
        "epochs": 250,
        "batch_size": 128,
        "optimizer": "SGD",
        "learning_rate": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "lr_decay_epochs": [150, 200],
        "temperature": 4.0,
        "label_smoothing": 0.1,
        "ce_weight": 0.5,
        "kd_weight": 0.5,
        "cbam": False,
        "feature_loss": False,
    }
    if contract != expected:
        raise ValueError(f"Experiment contract mismatch: {contract}")
    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=[42, 2025], required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tsne-max-samples", type=int, default=3000)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    reference_root = args.reference_root.resolve()
    output_root = args.output_root.resolve()
    data_root = (args.data_root or output_root.parent / "data").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config_dir = output_root / "configs"
    log_dir = output_root / "logs"
    evaluation_dir = output_root / "final_evaluation"
    figure_dir = output_root / "figures"
    started_at = datetime.now(timezone.utc)

    reference_checkpoints = {
        "baseline": find_checkpoint(reference_root, args.seed, "baseline"),
        "teacher": find_checkpoint(reference_root, args.seed, "teacher"),
        "cbam_kd_student": find_checkpoint(reference_root, args.seed, "cbam_kd_student"),
    }
    config = configure(
        read_yaml(project_root / "configs/vanilla_kd.yaml"),
        args.seed,
        output_root / "vanilla_kd_student",
        data_root,
        args.num_workers,
    )
    config["teacher"]["checkpoint"] = str(reference_checkpoints["teacher"])
    config_path = config_dir / "vanilla_kd.yaml"
    write_yaml(config_path, config)
    contract = validate_contract(config)
    (output_root / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8"
    )

    python = sys.executable
    scripts = project_root / "scripts"
    status_path = output_root / "pipeline_status.json"
    try:
        run_logged(
            [
                python,
                str(scripts / "train_vanilla_kd.py"),
                "--config",
                str(config_path),
                "--teacher-checkpoint",
                str(reference_checkpoints["teacher"]),
            ],
            log_dir / "01_train_vanilla_kd.log",
            cwd=project_root,
        )

        evaluation_dir.mkdir(parents=True, exist_ok=True)
        evaluation_specs = [
            ("baseline", "resnet20", reference_checkpoints["baseline"]),
            ("teacher", "resnet56", reference_checkpoints["teacher"]),
            ("vanilla_kd_student", "resnet20", output_root / "vanilla_kd_student/best.pth"),
            ("cbam_kd_student", "resnet20", reference_checkpoints["cbam_kd_student"]),
        ]
        summary_rows = []
        checkpoint_manifest = {}
        for index, (name, model_name, checkpoint) in enumerate(evaluation_specs, start=2):
            metrics_path = evaluation_dir / f"{name}.json"
            run_logged(
                [
                    python,
                    str(scripts / "evaluate.py"),
                    "--config",
                    str(config_path),
                    "--model",
                    model_name,
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(metrics_path),
                ],
                log_dir / f"{index:02d}_evaluate_{name}.log",
                cwd=project_root,
            )
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            summary_rows.append({"seed": args.seed, "model": name, **metrics})
            checkpoint_manifest[name] = {
                "filename": checkpoint.name,
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256(checkpoint),
                "included_in_archive": name == "vanilla_kd_student",
            }

        with (evaluation_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
        (evaluation_dir / "summary.json").write_text(
            json.dumps(summary_rows, indent=2, sort_keys=True), encoding="utf-8"
        )

        common_tsne = [
            python,
            str(scripts / "visualize_kd_comparison.py"),
            "--config",
            str(config_path),
            "--baseline-checkpoint",
            str(reference_checkpoints["baseline"]),
            "--teacher-checkpoint",
            str(reference_checkpoints["teacher"]),
            "--vanilla-kd-checkpoint",
            str(output_root / "vanilla_kd_student/best.pth"),
            "--cbam-kd-checkpoint",
            str(reference_checkpoints["cbam_kd_student"]),
            "--seed",
            str(args.seed),
        ]
        run_logged(
            common_tsne
            + [
                "--max-samples",
                str(args.tsne_max_samples),
                "--output",
                str(figure_dir / "tsne_all_classes.png"),
            ],
            log_dir / "06_tsne_all_classes.log",
            cwd=project_root,
        )
        run_logged(
            common_tsne
            + [
                "--classes",
                ",".join(str(value) for value in range(10)),
                "--max-samples",
                "1000",
                "--output",
                str(figure_dir / "tsne_ten_classes.png"),
            ],
            log_dir / "07_tsne_ten_classes.log",
            cwd=project_root,
        )

        finished_at = datetime.now(timezone.utc)
        status = {
            "status": "complete",
            "seed": args.seed,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "duration_seconds": (finished_at - started_at).total_seconds(),
            "experiment_contract": contract,
            "checkpoints": checkpoint_manifest,
            "system": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
            },
        }
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    except BaseException as error:
        failure = {
            "status": "failed",
            "seed": args.seed,
            "started_at_utc": started_at.isoformat(),
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        status_path.write_text(json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
        raise
    finally:
        archive = output_root.parent / f"vanilla_kd_seed_{args.seed}_results"
        shutil.make_archive(str(archive), "zip", root_dir=output_root.parent, base_dir=output_root.name)
        print(f"Result archive: {archive}.zip", flush=True)


if __name__ == "__main__":
    main()
