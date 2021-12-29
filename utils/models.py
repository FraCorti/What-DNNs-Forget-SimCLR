import torch
from torch import nn


class WideResNet(nn.Module):
    def __init__(self, depth=16, widen_factor=2, dropout=0.0):
        super().__init__()
        self.base = torch.hub.load('AdeelH/WideResNet-pytorch:torch_hub', 'WideResNet', depth=depth, num_classes=10,
                                   widen_factor=widen_factor, dropRate=dropout)

    def forward(self, x):
        return self.base(x)


def get_wide_resnet(depth=16, widen_factor=2, dropout=0.0):
    return WideResNet(depth=depth, widen_factor=widen_factor, dropout=dropout)


