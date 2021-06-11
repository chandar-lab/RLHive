import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np


class NoisyLinear(nn.Module):
    """NoisyLinear Layer"""

    def __init__(self, in_dim, out_dim, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_dim
        self.out_features = out_dim
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_dim, in_dim))
        self.weight_sigma = nn.Parameter(torch.empty(out_dim, in_dim))
        self.register_buffer("weight_epsilon", torch.empty(out_dim, in_dim))
        self.bias_mu = nn.Parameter(torch.empty(out_dim))
        self.bias_sigma = nn.Parameter(torch.empty(out_dim))
        self.register_buffer("bias_epsilon", torch.empty(out_dim))
        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def sample_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, inp):
        if self.training:
            return F.linear(
                inp,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon,
            )
        else:
            return F.linear(inp, self.weight_mu, self.bias_mu)


class ComplexMLP(nn.Module):
    """MLP function approximator for Q-Learning."""

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_units=256,
        num_hidden_layers=1,
        noisy=False,
        dueling=False,
        sigma_init=0.5,
    ):
        super().__init__()

        self._noisy = noisy
        self._dueling = dueling
        self._sigma_init = sigma_init
        self._in_dim = np.prod(in_dim)
        self._hidden_units = hidden_units
        if self._dueling:
            num_hidden_layers = max(num_hidden_layers - 1, 2)
        self._num_hidden_layers = num_hidden_layers
        self._out_dim = out_dim
        self._atoms = 1
        self.init_networks()

    def init_networks(self):
        if self._noisy:
            self.input_layer = nn.Sequential(
                NoisyLinear(self._in_dim, self._hidden_units), nn.ReLU()
            )
            self.hidden_layers = nn.Sequential(
                *[
                    nn.Sequential(
                        NoisyLinear(self._hidden_units, self._hidden_units), nn.ReLU()
                    )
                    for _ in range(self._num_hidden_layers - 1)
                ]
            )
            self.output_layer = NoisyLinear(
                self._hidden_units, self._out_dim * self._atoms
            )

        else:
            self.input_layer = nn.Sequential(
                nn.Linear(self._in_dim, self._hidden_units), nn.ReLU()
            )
            self.hidden_layers = nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Linear(self._hidden_units, self._hidden_units), nn.ReLU()
                    )
                    for _ in range(self._num_hidden_layers - 1)
                ]
            )
            self.output_layer = nn.Linear(
                self._hidden_units, self._out_dim * self._atoms
            )

        if self._dueling:
            """In dueling, we have two heads - one for estimating advantage function and one for
            estimating value function. If `noisy` is also true, then each of these layers will
            be NoisyLinear()"""

            if self._noisy:

                self.output_layer_adv = nn.Sequential(
                    NoisyLinear(
                        self._hidden_units, self._hidden_units, self._sigma_init
                    ),
                    nn.ReLU(),
                    NoisyLinear(
                        self._hidden_units,
                        self._out_dim * self._atoms,
                        self._sigma_init,
                    ),
                )

                self.output_layer_val = nn.Sequential(
                    NoisyLinear(
                        self._hidden_units, self._hidden_units, self._sigma_init
                    ),
                    nn.ReLU(),
                    NoisyLinear(
                        self._hidden_units,
                        self._out_dim * self._atoms,
                        self._sigma_init,
                    ),
                )

            else:
                self.output_layer_adv = nn.Sequential(
                    nn.Linear(self._hidden_units, self._hidden_units, self._sigma_init),
                    nn.ReLU(),
                    nn.Linear(
                        self._hidden_units,
                        self._out_dim * self._atoms,
                        self._sigma_init,
                    ),
                )

                self.output_layer_val = nn.Sequential(
                    nn.Linear(self._hidden_units, self._hidden_units, self._sigma_init),
                    nn.ReLU(),
                    nn.Linear(
                        self._hidden_units,
                        self._out_dim * self._atoms,
                        self._sigma_init,
                    ),
                )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.input_layer(x)
        x = self.hidden_layers(x)

        if self._dueling:
            adv = self.output_layer_adv(x)
            val = self.output_layer_val(x)

            if len(adv.shape) == 1:
                x = val + adv - adv.mean(0)
            else:
                x = (
                    val
                    + adv
                    - adv.mean(1).unsqueeze(1).expand(x.shape[0], self._out_dim)
                )

        else:
            x = self.output_layer(x)

        return x

    def sample_noise(self):
        self.input_layer[0].sample_noise()
        for i in range(len(self.hidden_layers[0])):
            if str(self.hidden_layers[0][i]) == "NoisyLinear()":
                self.hidden_layers[0][i].sample_noise()

        if self._dueling:
            for i in range(len(self.output_layer_adv)):
                if str(self.output_layer_adv[i]) == "NoisyLinear()":
                    self.output_layer_adv[i].sample_noise()

            for i in range(len(self.output_layer_val)):
                if str(self.output_layer_val[i]) == "NoisyLinear()":
                    self.output_layer_val[i].sample_noise()

        else:
            self.output_layer.sample_noise()


class DistributionalMLP(ComplexMLP):
    """Distributional MLP function approximator for Q-Learning."""

    def __init__(
        self,
        in_dim,
        out_dim,
        supports,
        hidden_units=256,
        num_hidden_layers=1,
        noisy=True,
        dueling=True,
        sigma_init=0.5,
        atoms=51,
    ):
        super(ComplexMLP, self).__init__()

        self._noisy = noisy
        self._dueling = dueling
        self._sigma_init = sigma_init
        self._in_dim = in_dim[0]
        self._out_dim = out_dim
        self._atoms = atoms
        self._supports = supports
        self._hidden_units = hidden_units
        self._num_hidden_layers = num_hidden_layers
        self.init_networks()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.dist(x)
        x = torch.sum(x * self._supports, dim=2)
        return x

    def dist(self, x):

        x = self.input_layer(x)
        x = self.hidden_layers(x)

        if self._dueling:
            adv = self.output_layer_adv(x)
            adv = adv.view(-1, self._out_dim, self._atoms)
            val = self.output_layer_val(x)
            val = val.view(-1, 1, self._atoms)
            x = val + adv - adv.mean(dim=1).view(-1, 1, self._atoms)

        else:
            x = self.output_layer(x)

        x = x.view(-1, self._out_dim, self._atoms)
        x = F.softmax(x, dim=-1)
        x = x.clamp(min=1e-3)

        return x
