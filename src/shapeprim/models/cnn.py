"""CNN baseline: ResNet-18, from scratch or ImageNet-pretrained.

From scratch. Standard ResNet-18's first layers (7x7 stride-2 conv +
stride-2 maxpool) were designed for 224x224 ImageNet input; on our 64px
images they would throw away most of the spatial resolution before the
first residual block runs. The from-scratch model therefore swaps in a 3x3
stem and drops the maxpool (the standard "ResNet for small images"
adaptation). ``stem_stride`` picks that stem's stride: 1 keeps the full
64x64 map through layer1 (the original setting, ~1.7 s per 32-image batch
on 4 CPU cores), 2 halves it to 32x32 -- the CIFAR-ResNet operating point
-- for a ~5x cheaper step. Experiments from Phase 1's learning curves on
use stride 2; the configs name it explicitly.

Pretrained. Keeps the original stem the ImageNet weights were learned for,
resizes the input to ``input_size`` and applies ImageNet normalisation
inside the module, so the data pipeline is identical for every CNN.
``freeze_backbone`` turns it into a linear probe: only the final layer
trains and batch-norm statistics stay at their ImageNet values. That is
what a practitioner with ten examples per class would actually do, which
makes it the honest few-shot pixel baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PretrainedResNet(nn.Module):
    def __init__(self, num_classes: int, input_size: int = 128, freeze_backbone: bool = False):
        super().__init__()
        self.net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)
        self.input_size = input_size
        self.freeze_backbone = freeze_backbone
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        if freeze_backbone:
            for name, p in self.net.named_parameters():
                p.requires_grad = name.startswith("fc.")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            # A linear probe keeps ImageNet batch-norm statistics; letting
            # them drift on 50 images would quietly fine-tune the backbone.
            for m in self.net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return self.net((x - self.mean) / self.std)


def build_cnn(
    num_classes: int,
    pretrained: bool = False,
    stem_stride: int = 1,
    input_size: int = 128,
    freeze_backbone: bool = False,
) -> nn.Module:
    if pretrained:
        return PretrainedResNet(num_classes, input_size=input_size, freeze_backbone=freeze_backbone)
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=stem_stride, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
