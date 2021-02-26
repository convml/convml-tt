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


class Flatten(nn.Module):
    """Flatten `x` to a single dimension, often used at the end of a model.
    `full` for rank-1 tensor"""

    def __init__(self, full: bool = False):
        self.full = full

    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)
