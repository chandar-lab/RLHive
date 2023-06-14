from functools import partial
from typing import Optional, Sequence, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Discrete

from hive.utils.torch_utils import calculate_output_dim
from hive.types import Creates, Partial, default
from hive.utils.registry import registry
from hive.utils.torch_utils import ModuleInitFn


def actor_critic_init_fn(module, std=np.sqrt(2), bias_const=0.0):
    if type(module) == torch.nn.Linear:
        torch.nn.init.orthogonal_(module.weight, std)
        torch.nn.init.constant_(module.bias, bias_const)


class CategoricalHead(torch.nn.Module):
    """A module that implements a discrete actor head. It uses the ouput from
    the :obj:`actor_net`, and adds creates a
    :py:class:`~torch.distributions.categorical.Categorical` object to compute
    the action distribution."""

    def __init__(self, feature_dim: int, action_space: gym.spaces.Discrete) -> None:
        """
        Args:
            feature dim: Expected output shape of the actor network.
            action_shape: Expected shape of actions.
        """
        super().__init__()
        self.network = torch.nn.Linear(feature_dim, int(action_space.n))
        self.distribution = torch.distributions.categorical.Categorical

    def forward(self, x):
        logits = self.network(x)
        return self.distribution(logits=logits)


class GaussianPolicyHead(torch.nn.Module):
    """A module that implements a continuous actor head. It uses the output from the
    :obj:`actor_net` and state independent learnable parameter :obj:`policy_logstd` to
    create a :py:class:`~torch.distributions.normal.Normal`  object to compute
    the action distribution."""

    def __init__(self, feature_dim: int, action_space: gym.spaces.Box) -> None:
        """
        Args:
            feature dim: Expected output shape of the actor network.
            action_shape: Expected shape of actions.
        """
        super().__init__()
        self._action_shape = action_space.shape
        self.policy_mean = torch.nn.Linear(
            feature_dim, int(np.prod(self._action_shape))
        )
        self.policy_logstd = torch.nn.Parameter(
            torch.zeros(1, int(np.prod(action_space.shape)))
        )
        self.distribution = torch.distributions.normal.Normal

    def forward(self, x):
        _mean = self.policy_mean(x)
        _std = self.policy_logstd.repeat(x.shape[0], 1).exp()
        distribution = self.distribution(
            torch.reshape(_mean, (x.size(0), *self._action_shape)),
            torch.reshape(_std, (x.size(0), *self._action_shape)),
        )
        return distribution


class ActorCriticNetwork(torch.nn.Module):
    """A module that implements the actor and critic computation. It puts together
    the :obj:`representation_network`, :obj:`actor_net` and :obj:`critic_net`, then
    adds two final :py:class:`~torch.nn.Linear` layers to compute the action and state
    value."""

    def __init__(
        self,
        action_space: Union[Box, Discrete],
        representation_network: torch.nn.Module,
        network_output_dim: Union[int, Sequence[int]],
        actor_net: Optional[Creates[torch.nn.Module]] = None,
        critic_net: Optional[Creates[torch.nn.Module]] = None,
        actor_head_init_fn: Optional[Partial[ModuleInitFn]] = None,
        critic_head_init_fn: Optional[Partial[ModuleInitFn]] = None,
    ) -> None:
        super().__init__()
        self._network = representation_network
        self._continuous_action = isinstance(action_space, Box)
        actor_net = default(actor_net, torch.nn.Identity)
        actor_network = actor_net(network_output_dim)

        actor_head_init_fn = default(
            actor_head_init_fn, partial(actor_critic_init_fn, std=0.01)
        )

        feature_dim = np.prod(calculate_output_dim(actor_network, network_output_dim))  # type: ignore
        if isinstance(action_space, Box):
            actor_head = GaussianPolicyHead(feature_dim, action_space)
        else:
            actor_head = CategoricalHead(feature_dim, action_space)

        self.actor = torch.nn.Sequential(
            actor_network,
            torch.nn.Flatten(),
            actor_head.apply(actor_head_init_fn),
        )

        critic_net = default(critic_net, torch.nn.Identity)
        critic_network = critic_net(network_output_dim)

        critic_head_init_fn = default(
            critic_head_init_fn, partial(actor_critic_init_fn, std=1)
        )

        feature_dim = np.prod(calculate_output_dim(critic_network, network_output_dim))  # type: ignore
        self.critic = torch.nn.Sequential(
            critic_network,
            torch.nn.Flatten(),
            torch.nn.Linear(feature_dim, 1).apply(critic_head_init_fn),
        )

    def forward(self, x, action=None):
        hidden_state = self._network(x)
        distribution = self.actor(hidden_state)
        value = self.critic(hidden_state)
        if action is None:
            action = distribution.sample()

        logprob, entropy = distribution.log_prob(action), distribution.entropy()
        if self._continuous_action:
            logprob, entropy = logprob.sum(dim=-1), entropy.sum(dim=-1)
        return action, logprob, entropy, value


registry.register("actor_critic_init", actor_critic_init_fn, ModuleInitFn)
