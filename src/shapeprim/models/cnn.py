"""CNN baseline: ResNet-18, adapted for small images.

Standard ResNet-18's first layers (7x7 stride-2 conv + stride-2 maxpool)
were designed for 224x224 ImageNet input; on our 64-128px images they'd
throw away most of the spatial resolution before the first residual block
even runs. When training from scratch we swap in a 3x3 stride-1 stem and
drop the maxpool (the standard "ResNet for small images" adaptation).
Pretrained mode keeps the original stem, since the ImageNet weights were
learned for it, and expects 224x224 input.
"""

from __future__ import annotations

import torch.nn as nn
import torchvision


def build_cnn(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    if not pretrained:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
