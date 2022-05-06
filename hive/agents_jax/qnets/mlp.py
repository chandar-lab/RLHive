from functools import partial
from typing import List, Tuple, Union
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as onp
from dataclasses import dataclass, field


class JaxMLPNetwork(nn.Module):
    """Jax MLP network """
    min_vals: Union[None, Tuple[float, ...]] = None,
    max_vals: Union[None, Tuple[float, ...]] = None,
    hidden_units: list = field(default_factory=list),  # Union[int, List[int]] = 256
    in_dim: Tuple[int] = 5,
    # hidden_units: list = field(default_factory=list) #Union[int, List[int]] = 256
    noisy: bool = False,
    std_init: float = 0.5,

    # hidden_units: List = field(default_factory=list)  # Union[int, List[int]] = 256


    def setup(self):

        # super().__init__(rng)

        if isinstance(self.hidden_units, int):
            hidden_units = [self.hidden_units]


        self.num_layers = len(self.hidden_units)

        initializer = nn.initializers.xavier_uniform()  ## TODO add NoisyLinear for Jax

        self.network = [
            nn.Dense(features=self.hidden_units, kernel_init=initializer)
            for _ in range(self.num_layers)]


        # modules = [nn.Dense(features=self.in_dim, kernel_init=initializer)]
        #
        # for i in range(len(self.hidden_units) - 1):
        #     modules.append(
        #         nn.Dense(features=self.hidden_units[i], kernel_init=initializer)
        #     )
        #
        # self.network = modules


    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = x.reshape((-1))  # flatten
        for layer in self.network:
            x = layer(x)
            x = nn.relu(x)
        return x
