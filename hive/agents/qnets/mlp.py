from functools import partial
from hive.agents.qnets.noisy_linear import NoisyLinear
import numpy as np
import torch
from torch import nn


class MLPNetwork(nn.Module):
    """Simple MLP function approximator for Q-Learning."""

    def __init__(self, in_dim, hidden_units=256, noisy=False, std_init=0.5):
        super().__init__()
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]
        modules = [torch.nn.Linear(np.prod(in_dim), hidden_units[0])]
        linear_fn = partial(NoisyLinear, std_init=std_init) if noisy else nn.Linear
        for i in range(len(hidden_units) - 1):
            modules.append(torch.nn.ReLU())
            modules.append(linear_fn(hidden_units[i], hidden_units[i + 1]))
        self.network = torch.nn.Sequential(*modules)

    def forward(self, x):
        x = x.float()
        x = torch.flatten(x, start_dim=1)
        return self.network(x)
