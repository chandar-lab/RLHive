import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np


class ComplexHanabiMLP(nn.Module):
    """MLP function approximator for Q-Learning."""

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_units=256,
        num_hidden_layers=1,
        dueling=False,
        sigma_init=0.5,
        atoms=1,
    ):
        super().__init__()

        self._dueling = dueling
        self._sigma_init = sigma_init
        self._in_dim = np.prod(in_dim)
        self._hidden_units = hidden_units
        if self._dueling:
            num_hidden_layers = max(num_hidden_layers - 1, 2)
        self._num_hidden_layers = num_hidden_layers
        self._out_dim = out_dim
        self._atoms = atoms
        self.init_networks()

    def init_networks(self):
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

        if self._dueling:
            """In dueling, we have two heads - one for estimating advantage function and one for
            estimating value function."""

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
                    1 * self._atoms,
                    self._sigma_init,
                ),
            )
        else:
            self.output_layer = nn.Linear(
                self._hidden_units, self._out_dim * self._atoms
            )

    def forward(self, x, legal_moves):
        x = torch.flatten(x, start_dim=1).float()
        x = self.input_layer(x)
        x = self.hidden_layers(x)

        if self._dueling:
            adv = self.output_layer_adv(x) * legal_moves
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
            x = self.output_layer(x) * legal_moves

        return x


class DistributionalHanabiMLP(ComplexHanabiMLP):
    """Distributional MLP function approximator for Q-Learning."""

    def __init__(
        self,
        in_dim,
        out_dim,
        supports,
        hidden_units=256,
        num_hidden_layers=1,
        dueling=True,
        sigma_init=0.5,
        atoms=51,
    ):
        super().__init__(
            in_dim,
            out_dim,
            hidden_units,
            num_hidden_layers,
            dueling,
            sigma_init,
            atoms,
        )
        self._supports = supports

    def forward(self, x, legal_moves):
        x = torch.flatten(x, start_dim=1)
        x = self.dist(x, legal_moves)
        x = torch.sum(x * self._supports, dim=2)
        return x

    def dist(self, x, legal_moves):

        x = self.input_layer(x.float())
        x = self.hidden_layers(x)

        if self._dueling:
            adv = self.output_layer_adv(x) * legal_moves.repeat(
                1, self._atoms
            )
            adv = adv.view(-1, self._out_dim, self._atoms)
            val = self.output_layer_val(x)
            val = val.view(-1, 1, self._atoms)
            x = val + adv - adv.mean(dim=1, keepdim=True)

        else:
            x = self.output_layer(x) * legal_moves.repeat(1, self._atoms)

        x = x.view(-1, self._out_dim, self._atoms)
        x = F.softmax(x, dim=-1)
        x = x.clamp(min=1e-3)

        return x
