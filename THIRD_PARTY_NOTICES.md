# Third-party notices

## CIFAR ResNet implementation

The ResNet implementation in `src/cbam_kd/models/resnet_cifar.py` is based on
ideas and structure from the following projects:

- Yerlan Idelbayev, **Proper ResNet Implementation for CIFAR10/CIFAR100 in
  PyTorch**: <https://github.com/akamaster/pytorch_resnet_cifar10>
  License: BSD 2-Clause. A copy is included in
  `licenses/akamaster-BSD-2-Clause.txt`.

- TorchVision ResNet implementation, PyTorch contributors:
  <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>
  License: BSD 3-Clause. A copy is included in
  `licenses/torchvision-BSD-3-Clause.txt`.

The local implementation uses the CIFAR stage depths and channel widths, a
projection shortcut for downsampling, adaptive average pooling, and explicit
methods for returning intermediate features.

## Research references

The method also builds on the following published work:

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. *Deep Residual Learning
  for Image Recognition.* CVPR, 2016.
- Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. *Distilling the Knowledge in a
  Neural Network.* 2015.
- Sanghyun Woo, Jongchan Park, Joon-Young Lee, and In So Kweon. *CBAM:
  Convolutional Block Attention Module.* ECCV, 2018.

CIFAR-100 is not redistributed in this repository. It is downloaded through
TorchVision when the training or evaluation scripts are run.
