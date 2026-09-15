# Reproducibility notes

The modular code in this repository was reconstructed from the original
notebook, the thesis, and the saved model checkpoints. The original notebook is
kept in `notebooks/original_experiment.ipynb` for reference.

A few differences are important when comparing a new run with the historical
results.

## Issues found in the original notebook

1. The loss function expects student logits before teacher logits, but the
   training call passes them in the opposite order. This changes both the
   cross-entropy term and the direction of logit distillation.

2. The validation block computes new teacher and student feature dictionaries,
   but then applies CBAM to feature tensors left over from the training loop.

3. The CBAM modules are created with trainable parameters, but only the student
   model parameters are passed to the optimizer.

4. The official CIFAR-100 test loader is used repeatedly during training to
   select the best checkpoint. The modular code instead creates a validation
   subset from the training data and evaluates the test set only after model
   selection.

5. The written thesis and the notebook do not use exactly the same training
   hyperparameters. The thesis describes 250 epochs, an initial learning rate of
   0.1, and stepwise learning-rate decay. The notebook contains a 400-epoch run
   with a 0.001 initial learning rate and cosine annealing.

## Current repository behavior

The scripts in `scripts/` use explicit student/teacher argument names, current
validation features, a held-out validation split, and include the distillation
CBAM parameters in the optimizer. The configuration
`configs/thesis_reconstruction.yaml` follows the written thesis description as
closely as the available records allow.

The historical checkpoints are preserved unchanged. In particular,
`resnet20_kd_best.pth` contains only the ResNet-20 student weights. It does not
store the CBAM weights, optimizer state, scheduler state, random split indices,
or other run metadata. For that reason, the original training run cannot be
resumed exactly from the supplied checkpoint.

The original accuracy values are therefore labeled as thesis-reported results.
Two clean 250-epoch runs of the current pipeline (seeds 42 and 2025) are reported
separately in the README and `results/reproduced_results.csv`; they do not
retroactively validate the exact historical training procedure.
