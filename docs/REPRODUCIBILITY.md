# Reproducibility notes

This repository contains the maintained implementation used for the thesis experiments. The modular code in `src/` and `scripts/` is the primary implementation, while the earlier exploratory notebook is retained at `notebooks/original_experiment.ipynb` for reference.

## Experiment pipeline

The reported experiments use a ResNet-56 teacher and ResNet-20 student on CIFAR-100. The official 50,000-image training set is divided into 45,000 training images and 5,000 validation images using a fixed split seed. The official 10,000-image test set is reserved for final evaluation after checkpoint selection.

The main CBAM-KD configuration is stored in:

```text
configs/cbam_kd.yaml
```

The teacher is trained first and its best validation checkpoint is then supplied to the CBAM-KD training script.

## Implementation notes

During development, the exploratory notebook and the final modular pipeline differed in several important implementation details. The maintained pipeline uses the following behavior:

1. Student and teacher logits are passed explicitly to the distillation loss in the correct order.

2. Validation uses feature tensors produced by the current validation batch.

3. The trainable CBAM parameters are included in the optimizer together with the student parameters.

4. Checkpoint selection is based on a held-out validation subset rather than repeated evaluation on the official test set.

5. The reported experiments use 250 epochs, an initial learning rate of 0.1, and MultiStepLR decay at epochs 150 and 200.

## Checkpoints

Model checkpoints are not committed to Git. A standard training sequence is:

```bash
python scripts/train_baseline.py --config configs/resnet20_baseline.yaml
python scripts/train_baseline.py --config configs/resnet56_teacher.yaml
python scripts/train_kd.py \
  --config configs/cbam_kd.yaml \
  --teacher-checkpoint runs/resnet56_teacher/best.pth
```

The checkpoint saved by the modular training scripts contains the model state and experiment metadata needed for evaluation. Large `.pth`, `.pt`, and `.ckpt` files remain excluded through `.gitignore`.

## Reported runs

The main results reported in the README were obtained from two complete 250-epoch runs using optimization seeds 42 and 2025 with a fixed train/validation split. The corresponding summary values are stored in `results/main_results.csv`.

Because only two optimization seeds are reported, the resulting standard deviations should be interpreted as descriptive rather than as a precise estimate of population-level training variance.
