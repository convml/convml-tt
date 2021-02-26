"""
Functionality from fastai used in model architecture
"""
import torch
from torch import nn


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"

    def __init__(self, size=1):
        super().__init__()
        self.size = size
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d(self.size)
        self.adaptive_maxpool = nn.AdaptiveMaxPool2d(self.size)

    def forward(self, x):
        return torch.cat([self.adaptive_maxpool(x), self.adaptive_avgpool(x)], 1)
