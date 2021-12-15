import math

import torch
import torch.nn.functional as F
from torch import nn


class NoisyLinear(nn.Module):
    """NoisyLinear Layer. Implements the layer described in
    https://arxiv.org/abs/1706.10295."""

    def __init__(self, in_dim: int, out_dim: int, std_init: float = 0.5):
        """
        Args:
            in_dim (int): The dimension of the input.
            out_dim (int): The desired dimension of the output.
            std_init (float): The range for the initialization of the standard deviation of the
                weights.
        """
        super().__init__()
        self.in_features = in_dim
        self.out_features = out_dim
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_dim, in_dim))
        self.weight_sigma = nn.Parameter(torch.empty(out_dim, in_dim))
        self.register_buffer("weight_epsilon", torch.empty(out_dim, in_dim))
        self.bias_mu = nn.Parameter(torch.empty(out_dim))
        self.bias_sigma = nn.Parameter(torch.empty(out_dim))
        self.register_buffer("bias_epsilon", torch.empty(out_dim))
        self._reset_parameters()
        self._sample_noise()

    def _reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign() * (x.abs().sqrt())

    def _sample_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        weight_eps = epsilon_out.ger(epsilon_in)
        bias_eps = epsilon_out
        return weight_eps, bias_eps

    def forward(self, inp):
        if self.training:
            weight_eps, bias_eps = self._sample_noise()
            return F.linear(
                inp,
                self.weight_mu
                + self.weight_sigma * weight_eps.to(device=self.weight_sigma.device),
                self.bias_mu
                + self.bias_sigma * bias_eps.to(device=self.bias_sigma.device),
            )
        else:
            return F.linear(inp, self.weight_mu, self.bias_mu)
