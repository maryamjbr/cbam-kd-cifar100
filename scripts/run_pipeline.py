#!/usr/bin/env python3
"""Run the complete baseline, teacher, CBAM-KD, evaluation, and t-SNE pipeline."""

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


def configure(
    base: dict[str, Any],
    *,
    seed: int,
    output_dir: Path,
    data_root: Path,
    workers: int,
    epochs: int | None,
) -> dict[str, Any]:
    config = json.loads(json.dumps(base))
    config["experiment"]["seed"] = seed
    config["experiment"]["output_dir"] = str(output_dir)
    config["data"]["root"] = str(data_root)
    config["data"]["num_workers"] = workers
    config["data"]["download"] = True
    if epochs is not None:
        config["training"]["epochs"] = epochs
        config["training"]["milestones"] = [max(1, int(epochs * 0.6)), max(2, int(epochs * 0.8))]
    return config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=None, help="Override all epoch counts")
    parser.add_argument("--tsne-max-samples", type=int, default=3000)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    data_root = (args.data_root or output_root.parent / "data").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config_dir = output_root / "configs"
    log_dir = output_root / "logs"
    evaluation_dir = output_root / "final_evaluation"
    figure_dir = output_root / "figures"
    started_at = datetime.now(timezone.utc)

    baseline_config = configure(
        read_yaml(project_root / "configs/resnet20_baseline.yaml"),
        seed=args.seed,
        output_dir=output_root / "baseline",
        data_root=data_root,
        workers=args.num_workers,
        epochs=args.epochs,
    )
    teacher_config = configure(
        read_yaml(project_root / "configs/resnet56_teacher.yaml"),
        seed=args.seed,
        output_dir=output_root / "teacher",
        data_root=data_root,
        workers=args.num_workers,
        epochs=args.epochs,
    )
    kd_config = configure(
        read_yaml(project_root / "configs/cbam_kd.yaml"),
        seed=args.seed,
        output_dir=output_root / "cbam_kd_student",
        data_root=data_root,
        workers=args.num_workers,
        epochs=args.epochs,
    )
    kd_config["teacher"]["checkpoint"] = str(output_root / "teacher/best.pth")
    kd_config["student"]["initial_checkpoint"] = None

    config_paths = {
        "baseline": config_dir / "resnet20_baseline.yaml",
        "teacher": config_dir / "resnet56_teacher.yaml",
        "kd": config_dir / "cbam_kd.yaml",
    }
    write_yaml(config_paths["baseline"], baseline_config)
    write_yaml(config_paths["teacher"], teacher_config)
    write_yaml(config_paths["kd"], kd_config)

    python = sys.executable
    scripts = project_root / "scripts"
    status_path = output_root / "pipeline_status.json"
    try:
        run_logged(
            [python, str(scripts / "train_baseline.py"), "--config", str(config_paths["baseline"])],
            log_dir / "01_baseline.log",
            cwd=project_root,
        )
        run_logged(
            [python, str(scripts / "train_baseline.py"), "--config", str(config_paths["teacher"])],
            log_dir / "02_teacher.log",
            cwd=project_root,
        )
        run_logged(
            [
                python,
                str(scripts / "train_kd.py"),
                "--config",
                str(config_paths["kd"]),
                "--teacher-checkpoint",
                str(output_root / "teacher/best.pth"),
            ],
            log_dir / "03_cbam_kd.log",
            cwd=project_root,
        )

        evaluation_dir.mkdir(parents=True, exist_ok=True)
        evaluation_specs = [
            ("baseline", "resnet20", config_paths["baseline"], output_root / "baseline/best.pth"),
            ("teacher", "resnet56", config_paths["teacher"], output_root / "teacher/best.pth"),
            ("cbam_kd_student", "resnet20", config_paths["kd"], output_root / "cbam_kd_student/best.pth"),
        ]
        summary_rows = []
        checkpoints: dict[str, dict[str, Any]] = {}
        for name, model_name, config_path, checkpoint in evaluation_specs:
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
                log_dir / f"04_evaluate_{name}.log",
                cwd=project_root,
            )
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            summary_rows.append({"seed": args.seed, "model": name, **metrics})
            checkpoints[name] = {
                "path": str(checkpoint.relative_to(output_root)),
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256(checkpoint),
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
            str(scripts / "visualize_tsne.py"),
            "--config",
            str(config_paths["kd"]),
            "--baseline-checkpoint",
            str(output_root / "baseline/best.pth"),
            "--teacher-checkpoint",
            str(output_root / "teacher/best.pth"),
            "--kd-checkpoint",
            str(output_root / "cbam_kd_student/best.pth"),
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
            log_dir / "05_tsne_all_classes.log",
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
            log_dir / "06_tsne_ten_classes.log",
            cwd=project_root,
        )

        finished_at = datetime.now(timezone.utc)
        status = {
            "status": "complete",
            "seed": args.seed,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "duration_seconds": (finished_at - started_at).total_seconds(),
            "checkpoints": checkpoints,
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
        archive = output_root.parent / f"cbam_kd_seed_{args.seed}_results"
        shutil.make_archive(
            str(archive),
            "zip",
            root_dir=output_root.parent,
            base_dir=output_root.name,
        )
        print(f"Result archive: {archive}.zip", flush=True)


if __name__ == "__main__":
    main()
