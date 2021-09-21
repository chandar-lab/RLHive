import torch
from torch import nn
import torch.nn.functional as F
from hive.agents.qnets.qnet_heads import (
    DQNNetwork,
    DuelingNetwork,
    DistributionalNetwork,
)


class HanabiDQNNetwork(DQNNetwork):
    def __init__(self, network, hidden_dim, out_dim, linear_fn=None):
        super().__init__(network, hidden_dim, out_dim, linear_fn)

    def forward(self, x, legal_moves):
        x = self.network(x)
        x = self.ouput_layer(x)

        # In the case of not having dueling or distributional heads
        if legal_moves is not None:
            x = x * legal_moves

        return x


class HanabiDuelingNetwork(DuelingNetwork):
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
        super().__init__(shared_network, hidden_dim, out_dim, linear_fn, atoms)

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
        if legal_moves is not None:
            x = x * legal_moves

        return x


class HanabiDistributionalNetwork(DistributionalNetwork):
    """Distributional MLP function approximator for Q-Learning."""

    def __init__(
        self,
        base_network: nn.Module,
        out_dim,
        vmin=0,
        vmax=200,
        atoms=51,
    ):
        super().__init__(base_network, out_dim, vmin, vmax, atoms)

    def forward(self, x, legal_moves):
        x = self.dist(x, legal_moves)
        x = torch.sum(x * self._supports, dim=2)
        x = x * legal_moves
        return x

    def dist(self, x, legal_moves):
        x = self.base_network(x, None)
        x = x.view(-1, self._out_dim, self._atoms)
        x = F.softmax(x, dim=-1)
        x = x.clamp(min=1e-3)
        return x
