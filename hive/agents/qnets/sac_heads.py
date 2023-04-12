from typing import Tuple, Union
import torch
import numpy as np
from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.utils import calculate_output_dim
import gymnasium as gym


MIN_LOG_STD = -5
MAX_LOG_STD = 2


class GaussianPolicyHead(torch.nn.Module):
    """A module that implements a continuous actor head. It takes in the output
    from the :obj:`actor_net`, and adds creates a normal distribution to compute
    the action distribution. It also adds a tanh layer to bound the output of
    the network to the action space. The forward method returns a sampled action
    from the distribution and the log probability of the action. It also returns
    a dummy value to keep a consistent interface with the discrete actor head.
    """

    def __init__(self, feature_dim: Tuple[int], action_space: gym.spaces.Box) -> None:
        """
        Args:
            feature dim: Expected output shape of the actor network.
            action_shape: Expected shape of actions.
        """
        super().__init__()
        self._action_shape = action_space.shape
        self._policy_mean = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, np.prod(self._action_shape))
        )
        self._policy_logstd = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, np.prod(self._action_shape))
        )
        self._distribution = torch.distributions.normal.Normal

    def forward(self, x):
        mean = self._policy_mean(x)
        log_std = self._policy_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = MIN_LOG_STD + 0.5 * (MAX_LOG_STD - MIN_LOG_STD) * (log_std + 1)
        std = torch.exp(log_std)

        distribution = self._distribution(mean, std)
        unsquashed_action = distribution.rsample()
        action = torch.tanh(unsquashed_action)

        log_prob = distribution.log_prob(unsquashed_action)  # B x A
        log_prob = log_prob.sum(axis=1, keepdim=True)

        # Correct for Tanh squashing
        correction = 2 * (
            (
                np.log(2)
                - unsquashed_action
                - torch.nn.functional.softplus(-2 * unsquashed_action)
            )
        ).sum(axis=1, keepdim=True)
        log_prob = log_prob - correction
        log_prob = log_prob.sum(axis=1, keepdim=True)
        return action, log_prob, None


class CategoricalPolicyHead(torch.nn.Module):
    """A module that implements a discrete actor head. It uses the ouput from
    the :obj:`actor_net`, and adds creates a
    :py:class:`~torch.distributions.categorical.Categorical` object to compute
    the action distribution."""

    def __init__(self, input_shape: Tuple[int], num_actions: int) -> None:
        """
        Args:
            input_shape: Expected output shape of the actor network.
            num_actions: Number of actions.
        """
        super().__init__()
        self._network = torch.nn.Linear(input_shape, num_actions)
        self._distribution = torch.distributions.categorical.Categorical

    def forward(self, x):
        logits = self._network(x)
        dist = self._distribution(logits=logits)
        action = dist.sample()
        log_prob = torch.log_softmax(logits, dim=1)
        return action, log_prob, dist.probs


class SACActorNetwork(torch.nn.Module):
    """A module that implements the SAC actor computation. It puts together the
    :obj:`representation_network` and :obj:`actor_net`, and adds a final
    :py:class:`~torch.nn.Linear` layer to compute the action."""

    def __init__(
        self,
        representation_network: torch.nn.Module,
        actor_net: FunctionApproximator,
        representation_network_output_shape: Union[int, Tuple[int]],
        action_space: Union[gym.spaces.Box, gym.spaces.Discrete],
    ) -> None:
        """
        Args:
            representation_network (torch.nn.Module): Network that encodes the
                observations.
            actor_net (FunctionApproximator): Function that takes in the shape of the
                encoded observations and creates a network. This network takes the
                encoded observations from representation_net and outputs the
                representations used to compute the actions (ie everything except the
                last layer).
            network_output_shape: Expected output shape of the representation network.
            action_shape: Requiured shape of the output action.
        """
        super().__init__()

        self._action_shape = action_space.shape
        if actor_net is None:
            actor_network = torch.nn.Identity()
        else:
            actor_network = actor_net(representation_network_output_shape)
        feature_dim = np.prod(
            calculate_output_dim(actor_network, representation_network_output_shape)
        )
        actor_modules = [
            representation_network,
            actor_network,
            torch.nn.Flatten(),
        ]
        policy_head_fn = (
            GaussianPolicyHead
            if isinstance(action_space, gym.spaces.Box)
            else CategoricalPolicyHead
        )
        actor_modules.append(policy_head_fn(feature_dim, action_space))
        self.actor = torch.nn.Sequential(*actor_modules)

    def forward(self, x):
        return self.actor(x)


class SACContinuousCriticNetwork(torch.nn.Module):
    def __init__(
        self,
        representation_network: torch.nn.Module,
        critic_net: FunctionApproximator,
        network_output_shape: Union[int, Tuple[int]],
        action_space: Union[gym.spaces.Box, gym.spaces.Discrete],
        n_critics: int = 2,
    ) -> None:
        """
        Args:
            representation_network (torch.nn.Module): Network that encodes the
                observations.
            critic_net (FunctionApproximator): Function that takes in the shape of the
                encoded observations and creates a network. This network takes two
                inputs: the encoded observations from representation_net and actions.
                It outputs the representations used to compute the values of the
                actions (ie everything except the last layer).
            network_output_shape: Expected output shape of the representation network.
            action_space: Expected shape of actions.
            n_critics: How many copies of the critic to create. They will all use the
                shared representation from the representation_network.
        """
        super().__init__()
        self.network = representation_network
        if critic_net is None:
            critic_net = lambda x: torch.nn.Identity()
        self._n_critics = n_critics
        input_shape = (np.prod(network_output_shape) + np.prod(action_space.shape),)
        critics = [critic_net(input_shape) for _ in range(n_critics)]
        feature_dim = np.prod(calculate_output_dim(critics[0], input_shape=input_shape))
        self._critics = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    critic,
                    torch.nn.Flatten(),
                    torch.nn.Linear(feature_dim, 1),
                )
                for critic in critics
            ]
        )

    def forward(self, obs, actions):
        obs = self.network(obs)
        obs = torch.flatten(obs, start_dim=1)
        actions = torch.flatten(actions, start_dim=1)
        x = torch.cat([obs, actions], dim=1)
        return [critic(x) for critic in self._critics]


class SACDiscreteCriticNetwork(torch.nn.Module):
    def __init__(
        self,
        representation_network: torch.nn.Module,
        critic_net: FunctionApproximator,
        network_output_shape: Union[int, Tuple[int]],
        action_space: gym.spaces.Discrete,
        n_critics: int = 2,
    ) -> None:
        """
        Args:
            representation_network (torch.nn.Module): Network that encodes the
                observations.
            critic_net (FunctionApproximator): Function that takes in the shape of the
                encoded observations and creates a network. This network takes two
                inputs: the encoded observations from representation_net and actions.
                It outputs the representations used to compute the values of the
                actions (ie everything except the last layer).
            network_output_shape: Expected output shape of the representation network.
            action_space: Expected shape of actions.
            n_critics: How many copies of the critic to create. They will all use the
                shared representation from the representation_network.
        """
        super().__init__()
        self.network = representation_network
        if critic_net is None:
            critic_net = lambda x: torch.nn.Identity()
        self._n_critics = n_critics
        input_shape = np.prod(network_output_shape)
        critics = [critic_net(input_shape) for _ in range(n_critics)]
        feature_dim = np.prod(calculate_output_dim(critics[0], input_shape=input_shape))
        self._critics = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    critic,
                    torch.nn.Flatten(),
                    torch.nn.Linear(feature_dim, action_space.n),
                )
                for critic in critics
            ]
        )

    def forward(self, obs):
        obs = self.network(obs)
        obs = torch.flatten(obs, start_dim=1)
        return [critic(obs) for critic in self._critics]
