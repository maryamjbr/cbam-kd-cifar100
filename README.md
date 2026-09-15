# Feature-Based Knowledge Distillation with CBAM Attention

PyTorch implementation and reproducibility materials for a B.Sc. thesis on
knowledge distillation for CIFAR-100.

The project uses a ResNet-56 teacher and a ResNet-20 student. The complete
CBAM-KD objective combines supervised classification, temperature-scaled logit
distillation, and CBAM-guided matching of intermediate feature maps from the
three residual stages.

CBAM is used only in the feature-distillation branch during training. It is not
part of the deployed student network, so the final model remains a standard
ResNet-20 with no additional inference-time parameters.

![CBAM-KD framework](results/figures/framework.png)

## Highlights

- ResNet-56 teacher and ResNet-20 student on CIFAR-100.
- Vanilla logit KD and CBAM-guided feature distillation.
- Stage-wise feature matching at all three CIFAR ResNet stages.
- CBAM used only during training; the deployed student remains unchanged.
- Fixed 45,000/5,000 train/validation split with the official CIFAR-100 test set
  reserved for final evaluation.
- Modular training, evaluation, checkpoint inspection, and t-SNE scripts.
- Two repeated 250-epoch experiments with optimization seeds 42 and 2025.
- CPU smoke tests and GitHub Actions continuous integration.

## Results

### Reproduced experiments

The corrected training pipeline was evaluated on CIFAR-100 using two independent
training runs with seeds 42 and 2025. All models were trained for 250 epochs on
NVIDIA Tesla T4 GPUs.

The table reports final test Top-1 accuracy using the checkpoint with the best
validation accuracy.

| Seed | ResNet-20 baseline | ResNet-56 teacher | Vanilla KD | CBAM-KD |
|---:|---:|---:|---:|---:|
| 42 | 68.05% | 71.81% | 70.18% | 69.92% |
| 2025 | 68.25% | 71.33% | 70.17% | 70.56% |
| **Mean** | **68.15%** | **71.57%** | **70.18%** | **70.24%** |
| **Sample SD** | **0.14** | **0.34** | **0.01** | **0.45** |

Vanilla KD improves the ResNet-20 baseline by **2.03 percentage points** on
average, while CBAM-KD improves it by **2.09 percentage points**.

The difference between CBAM-KD and vanilla KD is small and inconsistent across
the two runs:

- Seed 42: CBAM-KD is **0.26 pp lower** than vanilla KD.
- Seed 2025: CBAM-KD is **0.39 pp higher** than vanilla KD.
- Mean difference: **+0.07 pp** in favor of CBAM-KD.

These results support the benefit of knowledge distillation over ordinary
ResNet-20 training in this setup. However, with only two runs and a difference
that changes direction across seeds, they do not provide evidence that
CBAM-guided feature matching consistently improves over vanilla logit
distillation.

Machine-readable results are available in
[`results/reproduced_results.csv`](results/reproduced_results.csv).

### Experimental note

The standalone ResNet-20 baseline and ResNet-56 teacher were trained with
standard cross-entropy. Both distilled students used label smoothing
(`epsilon = 0.1`).

The two distillation objectives were:

```text
Vanilla KD:
L = 0.5 * L_CE + 0.5 * L_KD

CBAM-KD:
L = 0.5 * L_CE + 0.2 * L_KD + 0.3 * L_feat
