from functools import partial
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn

from hive.agents.qnets.noisy_linear import NoisyLinear


class MLPNetwork(nn.Module):
    """Basic MLP neural network architecture.

    Contains a series of :py:class:`torch.nn.Linear` or
    :py:class:`~hive.agents.qnets.noisy_linear.NoisyLinear` layers, each of which
    is followed by a ReLU.
    """

    def __init__(
        self,
        in_dim: Tuple[int],
        hidden_units: Union[int, List[int]] = 256,
        noisy: bool = False,
        std_init: float = 0.5,
    ):
        """
        Args:
            in_dim (tuple[int]): The shape of input observations.
            hidden_units (int | list[int]): The number of neurons for each mlp layer.
            noisy (bool): Whether the MLP should use
                :py:class:`~hive.agents.qnets.noisy_linear.NoisyLinear` layers or normal
                :py:class:`torch.nn.Linear` layers.
            std_init (float): The range for the initialization of the standard deviation of the
                weights in :py:class:`~hive.agents.qnets.noisy_linear.NoisyLinear`.
        """
        super().__init__()
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]
        linear_fn = partial(NoisyLinear, std_init=std_init) if noisy else nn.Linear
        modules = [linear_fn(np.prod(in_dim), hidden_units[0]), torch.nn.ReLU()]
        for i in range(len(hidden_units) - 1):
            modules.append(linear_fn(hidden_units[i], hidden_units[i + 1]))
            modules.append(torch.nn.ReLU())
        self.network = torch.nn.Sequential(*modules)

    def forward(self, x):
        x = x.float()
        x = torch.flatten(x, start_dim=1)
        return self.network(x)
