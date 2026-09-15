# Historical checkpoint manifest

The historical model files are intentionally not stored in Git. If obtained
separately, place them in this directory using the filenames below. They are
bare PyTorch `state_dict` objects and load exactly into the projection-shortcut CIFAR ResNet
implementation in this repository.

| File | Architecture | Reported role | SHA-256 |
|---|---|---|---|
| `resnet56_cifar100_best.pth` | ResNet-56 | Teacher | `73edefa1d5b5d4eb139763cb6b0563dce0ab7110bbd87e18c2965d601a2ac1f9` |
| `resnet20_cifar100_best.pth` | ResNet-20 | Student baseline | `075f2c7f237f168cc6faaf9c50d96821f7bc37eb9a2a0e9bccaadeb53d639428` |
| `resnet20_kd_best.pth` | ResNet-20 | CBAM-KD student | `8ed42840d6bc905adf66d1f02e325082bae7669523a5c9e118e9b60f34fd497f` |

The files do not contain training configuration, CBAM state, optimizer state, or
validation-split metadata. The modular pipeline can train fresh checkpoints from
the versioned configurations instead.
