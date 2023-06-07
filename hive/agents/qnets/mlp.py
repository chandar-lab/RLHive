from functools import partial
from typing import Union, Optional, Callable
from collections.abc import Sequence
import numpy as np
import torch
from torch import nn

from hive.agents.qnets.noisy_linear import NoisyLinear
from hive.utils.utils import ActivationFn
from hive.agents.qnets.utils import InitializationFn
from hive.utils.registry import Creates, OCreates, default


class MLPNetwork(nn.Module):
    """Basic MLP neural network architecture.

    Contains a series of :py:class:`torch.nn.Linear` or
    :py:class:`~hive.agents.qnets.noisy_linear.NoisyLinear` layers, each of which
    is followed by a ReLU.
    """

    def __init__(
        self,
        in_dim: Union[int, Sequence[int]],
        hidden_units: Union[int, Sequence[int]] = 256,
        activation_fn: OCreates[nn.Module] = None,
        noisy: bool = False,
        std_init: float = 0.5,
        initialization_fn: OCreates[None] = None,
    ):
        """
        Args:
            in_dim (Sequence[int]): The shape of input observations.
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

        activation_fn = default(activation_fn, torch.nn.ReLU)

        linear_fn = partial(NoisyLinear, std_init=std_init) if noisy else nn.Linear
        modules = [linear_fn(int(np.prod(in_dim)), hidden_units[0]), activation_fn()]
        for i in range(len(hidden_units) - 1):
            modules.append(linear_fn(hidden_units[i], hidden_units[i + 1]))
            modules.append(activation_fn())

        if initialization_fn is not None:
            self.network = torch.nn.Sequential(*modules).apply(initialization_fn)

    def forward(self, x):
        x = x.float()
        x = torch.flatten(x, start_dim=1)
        return self.network(x)
