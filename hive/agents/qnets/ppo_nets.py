from typing import Tuple, Union
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import torch

from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.utils import calculate_output_dim


class CategoricalHead(torch.nn.Module):
    """A module that implements a discrete actor head. It uses the ouput from the
    :obj:`actor_net`, and adds creates a :py:class:`~torch.distributions.categorical.Categorical`
    object to compute the action distribution."""

    def __init__(
        self, feature_dim: Tuple[int], action_space: gym.spaces.Discrete
    ) -> None:
        """
        Args:
            feature dim: Expected output shape of the actor network.
            action_shape: Expected shape of actions.
        """
        super().__init__()
        self.network = torch.nn.Linear(feature_dim, action_space.n)
        self.distribution = torch.distributions.categorical.Categorical

    def forward(self, x):
        logits = self.network(x)
        return self.distribution(logits=logits)


class GaussianPolicyHead(torch.nn.Module):
    """A module that implements a continuous actor head. It uses the output from the
    :obj:`actor_net` and state independent learnable parameter :obj:`policy_logstd` to
    create a :py:class:`~torch.distributions.normal.Normal`  object to compute
    the action distribution."""

    def __init__(self, feature_dim: Tuple[int], action_space: gym.spaces.Box) -> None:
        """
        Args:
            feature dim: Expected output shape of the actor network.
            action_shape: Expected shape of actions.
        """
        super().__init__()
        self._action_shape = action_space.shape
        self.policy_mean = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, np.prod(self._action_shape))
        )
        self.policy_logstd = torch.nn.Parameter(
            torch.zeros(1, np.prod(action_space.shape))
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


class PPOActorCriticNetwork(torch.nn.Module):
    """A module that implements the PPO actor and critic computation. It puts together the
    :obj:`representation_network`, :obj:`actor_net` and :obj:`critic_net`, then adds two final
    :py:class:`~torch.nn.Linear` layers to compute the action and state value."""

    def __init__(
        self,
        representation_network: torch.nn.Module,
        actor_net: FunctionApproximator,
        critic_net: FunctionApproximator,
        network_output_dim: Union[int, Tuple[int]],
        action_space: Union[Box, Discrete],
        continuous_action: bool,
    ) -> None:
        super().__init__()
        self._network = representation_network
        self._continuous_action = continuous_action
        if actor_net is None:
            actor_network = torch.nn.Identity()
        else:
            actor_network = actor_net(network_output_dim)
        feature_dim = np.prod(calculate_output_dim(actor_network, network_output_dim))
        actor_head = GaussianPolicyHead if self._continuous_action else CategoricalHead

        self.actor = torch.nn.Sequential(
            actor_network,
            torch.nn.Flatten(),
            actor_head(feature_dim, action_space),
        )

        if critic_net is None:
            critic_network = torch.nn.Identity()
        else:
            critic_network = critic_net(network_output_dim)
        feature_dim = np.prod(calculate_output_dim(critic_network, network_output_dim))
        self.critic = torch.nn.Sequential(
            critic_network,
            torch.nn.Flatten(),
            torch.nn.Linear(feature_dim, 1),
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
