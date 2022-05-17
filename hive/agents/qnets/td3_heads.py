import numpy as np
import torch

from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.utils import calculate_output_dim


class TD3ActorNetwork(torch.nn.Module):
    """A module that implements the TD3 actor computation. It puts together the
    :obj:`representation_network` and :obj:`actor_net`, and adds a final
    :py:class:`~torch.nn.Linear` layer to compute the action."""

    def __init__(
        self,
        representation_network: torch.nn.Module,
        actor_net: FunctionApproximator,
        network_output_shape,
        action_shape,
        use_tanh=True,
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

        self._action_shape = action_shape
        if actor_net is None:
            actor_network = torch.nn.Identity()
        else:
            actor_network = actor_net(network_output_shape)
        feature_dim = np.prod(calculate_output_dim(actor_network, network_output_shape))
        self.actor = torch.nn.Sequential(
            representation_network,
            actor_network,
            torch.nn.Flatten(),
            torch.nn.Linear(feature_dim, np.prod(action_shape)),
        )
        if use_tanh:
            self.actor.append(torch.nn.Tanh())

    def forward(self, x):
        x = self.actor(x)
        return torch.reshape(x, (x.size(0), *self._action_shape))


class TD3CriticNetwork(torch.nn.Module):
    def __init__(
        self,
        representation_network: torch.nn.Module,
        critic_net: FunctionApproximator,
        network_output_shape,
        n_critics: int,
        action_shape,
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
            n_critics: How many copies of the critic to create. They will all use the
                shared representation from the representation_network.
            action_shape: Expected shape of actions.
        """
        super().__init__()
        self.network = representation_network
        if critic_net is None:
            critic_net = lambda x: torch.nn.Identity()
        self._n_critics = n_critics
        input_shape = (np.prod(network_output_shape) + np.prod(action_shape),)
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
        if self._n_critics == 1:
            return [self._critics[0](x)]
        else:
            return [critic(x) for critic in self._critics]

    def q1(self, obs, actions):
        """Returns the value according to only the first critic."""
        obs = self.network(obs)
        x = torch.cat([obs, actions], dim=1)
        return self._critics[0](x)
