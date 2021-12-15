import torch
import torch.nn.functional as F
from torch import nn


class DQNNetwork(nn.Module):
    """Implements the standard DQN value computation. Transforms output from
    :obj:`base_network` with output dimension :obj:`hidden_dim` to dimension
    :obj:`out_dim`, which should be equal to the number of actions.
    """

    def __init__(
        self,
        base_network: nn.Module,
        hidden_dim: int,
        out_dim: int,
        linear_fn: nn.Module = None,
    ):
        """
        Args:
            base_network (torch.nn.Module): Backbone network that computes the
                representations that are used to compute action values.
            hidden_dim (int): Dimension of the output of the :obj:`network`.
            out_dim (int): Output dimension of the DQN. Should be equal to the
                number of actions that you are computing values for.
            linear_fn (torch.nn.Module): Function that will create the
                :py:class:`torch.nn.Module` that will take the output of
                :obj:`network` and produce the final action values. If
                :obj:`None`, a :py:class:`torch.nn.Linear` layer will be used.
        """
        super().__init__()
        self.base_network = base_network
        self._linear_fn = linear_fn if linear_fn is not None else nn.Linear
        self.output_layer = self._linear_fn(hidden_dim, out_dim)

    def forward(self, x):
        x = self.base_network(x)
        x = x.flatten(start_dim=1)
        return self.output_layer(x)


class DuelingNetwork(nn.Module):
    """Computes action values using Dueling Networks (https://arxiv.org/abs/1511.06581).
    In dueling, we have two heads---one for estimating advantage function and one for
    estimating value function.
    """

    def __init__(
        self,
        base_network: nn.Module,
        hidden_dim: int,
        out_dim: int,
        linear_fn: nn.Module = None,
        atoms: int = 1,
    ):
        """
        Args:
            base_network (torch.nn.Module): Backbone network that computes the
                representations that are shared by the two estimators.
            hidden_dim (int): Dimension of the output of the :obj:`base_network`.
            out_dim (int): Output dimension of the Dueling DQN. Should be equal
                to the number of actions that you are computing values for.
            linear_fn (torch.nn.Module): Function that will create the
                :py:class:`torch.nn.Module` that will take the output of
                :obj:`network` and produce the final action values. If
                :obj:`None`, a :py:class:`torch.nn.Linear` layer will be used.
            atoms (int): Multiplier for the dimension of the output. For standard
                dueling networks, this should be 1. Used by
                :py:class:`~hive.agents.qnets.qnet_heads.DistributionalNetwork`.
        """
        super().__init__()
        self.base_network = base_network
        self._hidden_dim = hidden_dim
        self._out_dim = out_dim
        self._atoms = atoms
        self._linear_fn = linear_fn if linear_fn is not None else nn.Linear
        self.init_networks()

    def init_networks(self):
        self.output_layer_adv = self._linear_fn(
            self._hidden_dim, self._out_dim * self._atoms
        )

        self.output_layer_val = self._linear_fn(self._hidden_dim, 1 * self._atoms)

    def forward(self, x):
        x = self.base_network(x)
        x = x.flatten(start_dim=1)
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
        return x


class DistributionalNetwork(nn.Module):
    """Computes a categorical distribution over values for each action
    (https://arxiv.org/abs/1707.06887)."""

    def __init__(
        self,
        base_network: nn.Module,
        out_dim: int,
        vmin: float = 0,
        vmax: float = 200,
        atoms: int = 51,
    ):
        """
        Args:
            base_network (torch.nn.Module): Backbone network that computes the
                representations that are used to compute the value distribution.
            out_dim (int): Output dimension of the Distributional DQN. Should be
                equal to the number of actions that you are computing values for.
            vmin (float): The minimum of the support of the categorical value
                distribution.
            vmax (float): The maximum of the support of the categorical value
                distribution.
            atoms (int): Number of atoms discretizing the support range of the
                categorical value distribution.
        """

        super().__init__()
        self.base_network = base_network
        self._supports = torch.nn.Parameter(torch.linspace(vmin, vmax, atoms))
        self._out_dim = out_dim
        self._atoms = atoms

    def forward(self, x):
        x = self.dist(x)
        x = torch.sum(x * self._supports, dim=2)
        return x

    def dist(self, x):
        """Computes a categorical distribution over values for each action."""
        x = self.base_network(x)
        x = x.view(-1, self._out_dim, self._atoms)
        x = F.softmax(x, dim=-1)
        return x
