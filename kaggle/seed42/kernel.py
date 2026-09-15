"""Kaggle entrypoint for the complete CBAM-KD CIFAR-100 seed-42 run."""

from pathlib import Path
import shutil
import subprocess
import sys

import torch


SEED = 42


if not torch.cuda.is_available():
    raise RuntimeError("This pipeline requires a Kaggle GPU session")
capability = torch.cuda.get_device_capability(0)
print(f"GPU: {torch.cuda.get_device_name(0)}, compute capability: {capability}", flush=True)
if capability < (7, 0):
    raise RuntimeError("The current Kaggle PyTorch image requires GPU compute capability 7.0+")


def find_project_root() -> Path:
    direct_candidates = sorted(Path("/kaggle/input").glob("**/pyproject.toml"))
    for candidate in direct_candidates:
        text = candidate.read_text(encoding="utf-8")
        if 'name = "cbam-kd-cifar100"' in text:
            project_root = Path("/kaggle/working/project_source")
            shutil.copytree(candidate.parent, project_root, dirs_exist_ok=True)
            return project_root

    archives = sorted(Path("/kaggle/input").glob("**/cbam_kd_source.zip"))
    if len(archives) != 1:
        raise FileNotFoundError(f"Expected one source archive, found {len(archives)}")
    extraction_root = Path("/kaggle/working/project_source")
    shutil.unpack_archive(archives[0], extraction_root)
    candidates = sorted(extraction_root.glob("**/pyproject.toml"))
    for candidate in candidates:
        text = candidate.read_text(encoding="utf-8")
        if 'name = "cbam-kd-cifar100"' in text:
            return candidate.parent
    raise FileNotFoundError("Could not find the CBAM-KD project in the source archive")


project_root = find_project_root()
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", str(project_root), "--no-deps"],
    check=True,
)
subprocess.run(
    [
        sys.executable,
        str(project_root / "scripts/run_pipeline.py"),
        "--seed",
        str(SEED),
        "--project-root",
        str(project_root),
        "--output-root",
        f"/kaggle/working/seed_{SEED}",
        "--data-root",
        "/kaggle/working/data",
        "--num-workers",
        "4",
    ],
    check=True,
)
