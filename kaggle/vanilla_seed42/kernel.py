"""Kaggle entrypoint for logit-only vanilla KD, seed 42."""

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


def unpack_project() -> Path:
    direct_candidates = sorted(Path("/kaggle/input").glob("**/pyproject.toml"))
    for candidate in direct_candidates:
        if 'name = "cbam-kd-cifar100"' in candidate.read_text(encoding="utf-8"):
            project_root = Path("/kaggle/working/project_source")
            shutil.copytree(candidate.parent, project_root, dirs_exist_ok=True)
            return project_root
    archives = sorted(Path("/kaggle/input").glob("**/vanilla_kd_source.zip"))
    if len(archives) != 1:
        raise FileNotFoundError(f"Expected one vanilla source archive, found {archives}")
    extraction_root = Path("/kaggle/working/project_source")
    shutil.unpack_archive(archives[0], extraction_root)
    candidates = sorted(extraction_root.glob("**/pyproject.toml"))
    for candidate in candidates:
        if 'name = "cbam-kd-cifar100"' in candidate.read_text(encoding="utf-8"):
            return candidate.parent
    raise FileNotFoundError("Could not find the project in vanilla_kd_source.zip")


def unpack_reference() -> Path:
    direct = sorted(Path("/kaggle/input").glob(f"**/seed_{SEED}/teacher/best.pth"))
    if len(direct) == 1:
        return Path("/kaggle/input")
    expected = f"cbam_kd_seed_{SEED}_results.zip"
    archives = sorted(Path("/kaggle/input").glob(f"**/{expected}"))
    if len(archives) != 1:
        raise FileNotFoundError(f"Expected one {expected}, found {archives}")
    extraction_root = Path("/kaggle/working/reference_results")
    shutil.unpack_archive(archives[0], extraction_root)
    return extraction_root


project_root = unpack_project()
reference_root = unpack_reference()
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", str(project_root), "--no-deps"],
    check=True,
)
subprocess.run(
    [
        sys.executable,
        str(project_root / "scripts/run_vanilla_kd_pipeline.py"),
        "--seed",
        str(SEED),
        "--project-root",
        str(project_root),
        "--reference-root",
        str(reference_root),
        "--output-root",
        f"/kaggle/working/vanilla_kd_seed_{SEED}",
        "--data-root",
        "/kaggle/working/data",
        "--num-workers",
        "4",
    ],
    check=True,
)
