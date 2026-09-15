# Feature-Based Knowledge Distillation with CBAM Attention

PyTorch implementation and reproducibility materials for a B.Sc. thesis on knowledge distillation for CIFAR-100.

The project uses a ResNet-56 teacher and a ResNet-20 student. The complete CBAM-KD objective combines supervised classification, temperature-scaled logit distillation, and CBAM-guided matching of intermediate feature maps from the three residual stages.

CBAM is used only in the feature-distillation branch during training. It is not part of the deployed student network, so the final model remains a standard ResNet-20 with no additional inference-time parameters.

![CBAM-KD framework](results/figures/framework.png)

## Highlights

- ResNet-56 teacher and ResNet-20 student on CIFAR-100.
- Vanilla logit KD and CBAM-guided feature distillation.
- Stage-wise feature matching at all three CIFAR ResNet stages.
- CBAM used only during training; the deployed student remains unchanged.
- Fixed 45,000/5,000 train/validation split with the official CIFAR-100 test set reserved for final evaluation.
- Modular training, evaluation, checkpoint inspection, and t-SNE scripts.
- Two repeated 250-epoch experiments with optimization seeds 42 and 2025.
- CPU smoke tests and GitHub Actions continuous integration.

## Results

### Reproduced experiments

The corrected training pipeline was evaluated on CIFAR-100 using two independent training runs with seeds 42 and 2025. All models were trained for 250 epochs on an NVIDIA Tesla T4 GPU per run.

The table reports final test Top-1 accuracy using the checkpoint with the best validation accuracy.

| Seed | ResNet-20 baseline | ResNet-56 teacher  | CBAM-KD |
|---:|---:|---:|---:|---:|
| 42 | 68.05% | 71.81% |69.92% |
| 2025 | 68.25% | 71.33% |70.56% |
| **Mean** | **68.15%** | **71.57%** | **70.24%** |

Vanilla KD improves the ResNet-20 baseline by **2.03 percentage points** on average, while CBAM-KD improves it by **2.09 percentage points**.

The difference between CBAM-KD and vanilla KD is small and inconsistent across the two runs:

- Seed 42: CBAM-KD is **0.26 pp lower** than vanilla KD.
- Seed 2025: CBAM-KD is **0.39 pp higher** than vanilla KD.
- Mean difference: **+0.07 pp** in favor of CBAM-KD.

These results support the benefit of knowledge distillation over ordinary ResNet-20 training in this setup. However, with only two runs and a difference that changes direction across seeds, they do not provide evidence that CBAM-guided feature matching consistently improves over vanilla logit distillation.

Machine-readable results are available in [`results/reproduced_results.csv`](results/reproduced_results.csv).

### Experimental note

The standalone ResNet-20 baseline and ResNet-56 teacher were trained with standard cross-entropy. Both distilled students used label smoothing (`epsilon = 0.1`).

The two distillation objectives were:

```text
Vanilla KD:
L = 0.5 * L_CE + 0.5 * L_KD

CBAM-KD:
L = 0.5 * L_CE + 0.2 * L_KD + 0.3 * L_feat
```

Therefore, the vanilla-KD versus CBAM-KD comparison evaluates two complete distillation objectives with the same total teacher-guidance weight. It is not a strict one-variable ablation in which only CBAM is switched on or off.


### Historical thesis results

The original thesis experiments produced the following values before the training and validation pipeline was revised:

| Model | Role | Reported Top-1 |
|---|---|---:|
| ResNet-56 | Teacher | 72.55% |
| ResNet-20 | Student baseline | 68.03% |
| ResNet-20 CBAM-KD | Distilled student | 68.68% |

These values are retained only as a record of the original experiment. The corrected reruns above should be used for current comparisons.

Implementation differences between the original notebook and the revised pipeline are documented in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Feature-space visualization

t-SNE was applied to global-average-pooled features from the final residual stage. The figures include the ResNet-20 baseline, ResNet-56 teacher, vanilla-KD student, and CBAM-KD student.

![Full CIFAR-100 t-SNE comparison](results/figures/tsne_all_classes.png)

![Ten-class t-SNE comparison](results/figures/tsne_ten_classes.png)

Each model is projected independently. The plots are therefore intended for qualitative inspection of within-panel structure only; positions and distances should not be compared directly across panels.

## Repository layout

```text
configs/                  Versioned experiment configurations
src/cbam_kd/              Models, CBAM, losses, data, metrics, and training utilities
scripts/                  Training, evaluation, pipelines, and visualization CLIs
kaggle/                   Kaggle kernel entry points and metadata
notebooks/                Original experiment notebook and a compact demo
tests/                    CPU smoke tests
results/                  Result tables and selected figures
checkpoints/README.md      Optional historical checkpoint manifest
docs/REPRODUCIBILITY.md   Implementation differences and reproducibility notes
```

Datasets, model weights, logs, raw Kaggle downloads, archives, and generated thesis files are excluded from Git.

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/maryamjbr/cbam-kd-cifar100.git
cd cbam-kd-cifar100

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -e ".[visualization,dev]"
```

CIFAR-100 is downloaded automatically by TorchVision on first use.

## Quick verification

```bash
pytest -q
```

The tests cover model shapes and parameter counts, CBAM-KD gradients, and the vanilla-KD objective. Checkpoint compatibility is also checked when optional historical checkpoint files are available.

## Training

Train the ResNet-20 baseline and ResNet-56 teacher:

```bash
python scripts/train_baseline.py --config configs/resnet20_baseline.yaml
python scripts/train_baseline.py --config configs/resnet56_teacher.yaml
```

Train the CBAM-KD student:

```bash
python scripts/train_kd.py --config configs/thesis_reconstruction.yaml
```

The KD configuration expects a teacher checkpoint at:

```text
checkpoints/resnet56_cifar100_best.pth
```

by default. Edit `teacher.checkpoint` in the YAML configuration or pass `--teacher-checkpoint` to use a different checkpoint.

Run the complete baseline, teacher, CBAM-KD, evaluation, and visualization pipeline:

```bash
python scripts/run_pipeline.py \
  --seed 42 \
  --output-root runs/seed_42
```

For a quick end-to-end check:

```bash
python scripts/run_pipeline.py \
  --seed 42 \
  --epochs 1 \
  --output-root runs/smoke_test
```

Run the vanilla-KD experiment after producing the corresponding baseline and teacher checkpoints:

```bash
python scripts/run_vanilla_kd_pipeline.py \
  --seed 42 \
  --reference-root runs \
  --output-root runs/vanilla_seed_42
```

For the reported results, the complete procedure was repeated with seeds `42` and `2025` while keeping the train/validation split fixed.

## Evaluation and visualization

Evaluate a ResNet-20 checkpoint:

```bash
python scripts/evaluate.py \
  --config configs/resnet20_baseline.yaml \
  --model resnet20 \
  --checkpoint /path/to/best.pth
```

Inspect checkpoint compatibility and file hashes:

```bash
python scripts/inspect_checkpoints.py /path/to/checkpoint.pth
```

See the `--help` output of each script for the complete set of options.

## Distillation objectives

Let `z_s` and `z_t` denote student and teacher logits, `y` the ground-truth labels, and `F_s` and `F_t` their intermediate feature maps.

The temperature-scaled logit-distillation loss is:

```text
L_KD = T^2 * KL(
    softmax(z_t / T)
    ||
    softmax(z_s / T)
)
```

with:

```text
T = 4
```

The vanilla-KD objective is:

```text
L_vanilla = 0.5 * L_CE
          + 0.5 * L_KD
```

The complete CBAM-KD objective is:

```text
L_CBAM-KD = 0.5 * L_CE
          + 0.2 * L_KD
          + 0.3 * L_feat
```

`L_feat` is a weighted mean-squared error between CBAM-refined teacher and student features at the three residual stages. Relative stage weights are `1:2:3`, giving greater weight to deeper representations.

A separate CBAM block is used at each stage and shared between the teacher and student feature tensors for that stage. The teacher is frozen, and its attended feature is treated as a stop-gradient target.

The CBAM blocks are optimized only during training and are discarded afterward. The deployed model is therefore the original ResNet-20 backbone.

## Reproducibility notes

The original experiment notebook is retained at:

```text
notebooks/original_experiment.ipynb
```

During modularization, several issues were identified in the original pipeline, including:

- reversed student and teacher logits in one distillation-loss call;
- stale feature tensors during validation;
- use of the test set during model selection.

The revised pipeline uses explicit loss arguments, current validation features, a fixed held-out validation split, and final test evaluation only after checkpoint selection.

The historical checkpoints contain only model `state_dict`s. Optimizer, scheduler, CBAM, and data-split states from the original experiment cannot be fully reconstructed.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for further details.

## Limitations

The current reproduced comparison is based on two training seeds. This is enough to show that both distillation configurations outperform the local ResNet-20 baseline in both runs, but it is not enough to resolve the small difference between vanilla KD and CBAM-KD.

A stricter future ablation would:

1. keep the CE and logit-KD coefficients fixed while adding or removing only the CBAM feature branch;
2. include a ResNet-20 baseline trained with the same label smoothing used by the distilled students;
3. repeat the experiment over additional seeds.

## Attribution and citation

The CIFAR ResNet implementation follows the structure of Yerlan Idelbayev's [`pytorch_resnet_cifar10`](https://github.com/akamaster/pytorch_resnet_cifar10) and conventions from TorchVision's [`resnet.py`](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py).

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`licenses/`](licenses/) for licensing and attribution details.

If you use this work, cite the metadata in [`CITATION.cff`](CITATION.cff).

## License

Released under the [BSD 3-Clause License](LICENSE). Third-party components retain their original license terms.
