from functools import partial
from typing import List, Tuple, Union
from flax import linen as nn
import gin
import jax
import jax.numpy as jnp
import numpy as onp


@gin.configurable
class JaxMLPNetwork(nn.Module):
  """Jax MLP network """

  min_vals: Union[None, Tuple[float, ...]] = None
  max_vals: Union[None, Tuple[float, ...]] = None
  in_dim: Tuple[int]
  hidden_units: Union[int, List[int]] = 256
  noisy: bool = False
  std_init: float = 0.5


  if isinstance(hidden_units, int):
      hidden_units = [hidden_units]

  def setup(self):

    initializer = nn.initializers.xavier_uniform() ## TODO add NoisyLinear for Jax

    modules = [nn.Dense(features=self.in_dim, kernel_init=initializer)]

    for i in range(len(self.hidden_units) - 1):
      modules.append(nn.relu())
      modules.append(nn.Dense(features=self.hidden_units[i], kernel_init=initializer))

    self.network = modules


  def forward(self, x):
      x = x.astype(jnp.float32)
      x = x.reshape((-1))  # flatten
      for layer in self.network:
          x = layer(x)
      return x






