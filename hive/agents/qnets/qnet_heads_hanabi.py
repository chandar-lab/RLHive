import torch
from torch import nn
import torch.nn.functional as F


class HanabiHead(nn.Module):
    def __init__(self, qnet):
        super().__init__()
        self._qnet = qnet

    def forward(self, x, legal_moves):
        x = self._qnet(x)
        return x + legal_moves
