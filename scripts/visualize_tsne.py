#!/usr/bin/env python3
"""Generate t-SNE plots from independently loaded final model checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from cbam_kd.checkpoint import load_model_checkpoint
from cbam_kd.config import load_config, section
from cbam_kd.data import build_loaders
from cbam_kd.models.resnet_cifar import build_resnet
from cbam_kd.utils import resolve_device, seed_everything


@torch.no_grad()
def collect(
    model,
    loader,
    device,
    max_samples: int,
    included_classes: set[int] | None = None,
):
    model.eval()
    feature_rows = []
    labels_rows = []
    seen = 0
    for images, labels in loader:
        if included_classes is not None:
            keep = torch.tensor(
                [int(label) in included_classes for label in labels],
                dtype=torch.bool,
            )
            images = images[keep]
            labels = labels[keep]
            if labels.numel() == 0:
                continue
        images = images.to(device)
        _, features = model.forward_with_features(images)
        pooled = torch.mean(features[-1], dim=(-2, -1)).cpu()
        remaining = max_samples - seen
        feature_rows.append(pooled[:remaining])
        labels_rows.append(labels[:remaining])
        seen += min(remaining, labels.shape[0])
        if seen >= max_samples:
            break
    return torch.cat(feature_rows).numpy(), torch.cat(labels_rows).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--kd-checkpoint", required=True)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--output", default="results/figures/tsne_reproduced.png")
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--classes",
        default=None,
        help="Optional comma-separated CIFAR-100 class IDs to include",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    seed_everything(args.seed)
    config = load_config(args.config)
    data_cfg = section(config, "data")
    device = resolve_device(args.device)
    loaders = build_loaders(data_cfg)

    included_classes = None
    if args.classes:
        included_classes = {int(value.strip()) for value in args.classes.split(",")}
        if not included_classes:
            raise ValueError("--classes must contain at least one class ID")
        if min(included_classes) < 0 or max(included_classes) > 99:
            raise ValueError("CIFAR-100 class IDs must be between 0 and 99")

    model_specs = [
        ("ResNet-20 baseline", "resnet20", args.baseline_checkpoint),
        ("ResNet-20 CBAM-KD", "resnet20", args.kd_checkpoint),
    ]
    if args.teacher_checkpoint:
        model_specs.insert(1, ("ResNet-56 teacher", "resnet56", args.teacher_checkpoint))

    labels = None
    embeddings = []
    for title, model_name, checkpoint in model_specs:
        model = build_resnet(model_name, 100).to(device)
        load_model_checkpoint(model, checkpoint, map_location=device)
        features, model_labels = collect(
            model,
            loaders.test,
            device,
            args.max_samples,
            included_classes,
        )
        if labels is None:
            labels = model_labels
        elif not np.array_equal(labels, model_labels):
            raise RuntimeError("Feature collections used different sample orders")
        if features.shape[0] <= args.perplexity:
            raise ValueError("The selected sample count must be greater than perplexity")
        embedding = TSNE(
            n_components=2,
            perplexity=args.perplexity,
            init="pca",
            learning_rate="auto",
            random_state=args.seed,
        ).fit_transform(features)
        embeddings.append((title, embedding))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(embeddings), figsize=(7 * len(embeddings), 6))
    axes = np.atleast_1d(axes)
    color_map = "tab10" if included_classes and len(included_classes) <= 10 else "turbo"
    for axis, (title, embedding) in zip(axes, embeddings, strict=True):
        axis.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels,
            cmap=color_map,
            s=5,
            alpha=0.75,
        )
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
