import torch
from torch import nn
import torch.nn.functional as F


class HanabiDQNNetwork(nn.Module):
    def __init__(self, network, hidden_dim, out_dim, linear_fn=None):
        super().__init__()
        self.network = network
        self._linear_fn = linear_fn if linear_fn is not None else nn.Linear
        self.ouput_layer = self._linear_fn(hidden_dim, out_dim)

    def forward(self, x):
        x = self.network(x)
        x = self.ouput_layer(x)

        # In the case of not having dueling or distributional heads
        if x.size(-1) == legal_moves.size(0):
            x = x * legal_moves

        return x


class HanabiDuelingNetwork(nn.Module):
    """In dueling, we have two heads - one for estimating advantage function and one for
    estimating value function."""

    def __init__(
        self,
        shared_network,
        hidden_dim,
        out_dim,
        linear_fn=None,
        atoms=1,
    ):
        super().__init__()
        self.shared_network = shared_network
        self._hidden_dim = hidden_dim
        self._out_dim = out_dim
        self._atoms = atoms
        self._linear_fn = linear_fn if linear_fn is not None else nn.Linear
        self.init_networks()

    def init_networks(self):
        self.output_layer_adv = nn.Sequential(
            self._linear_fn(self._hidden_dim, self._hidden_dim),
            nn.ReLU(),
            self._linear_fn(self._hidden_dim, self._out_dim * self._atoms),
        )

        self.output_layer_val = nn.Sequential(
            self._linear_fn(self._hidden_dim, self._hidden_dim),
            nn.ReLU(),
            self._linear_fn(self._hidden_dim, 1 * self._atoms),
        )

    def forward(self, x, legal_moves):
        x = self.shared_network(x)

        adv = self.output_layer_adv(x)
        val = self.output_layer_val(x)

        if adv.dim() == 1:
            x = val + adv - adv.mean(0)
        else:
            adv = adv.reshape(adv.size(0), self._out_dim, self._atoms)
            val = val.reshape(val.size(0), 1, self._atoms)
            x = val + adv - adv.mean(dim=1, keepdim=True)
            if self._atoms == 1:
                x = x.squeeze(dim=2)

        # In the case of not having distributional head
        if x.size(-1) == legal_moves.size(0):
            x = x * legal_moves

        return x


class HanabiDistributionalNetwork(nn.Module):
    """Distributional MLP function approximator for Q-Learning."""

    def __init__(
        self,
        base_network: nn.Module,
        out_dim,
        vmin=0,
        vmax=200,
        atoms=51,
    ):
        super().__init__()
        self.base_network = base_network
        self._supports = torch.nn.Parameter(torch.linspace(vmin, vmax, atoms))
        self._out_dim = out_dim
        self._atoms = atoms

    def forward(self, x, legal_moves):
        x = self.dist(x, legal_moves)
        x = torch.sum(x * self._supports, dim=2)
        x = x * legal_moves
        return x

    def dist(self, x, legal_moves):
        x = self.base_network(x)
        x = x.view(-1, self._out_dim, self._atoms)
        x = F.softmax(x, dim=-1)
        x = x.clamp(min=1e-3)
        return x
