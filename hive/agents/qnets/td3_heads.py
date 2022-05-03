import numpy as np
import torch

from hive.agents.qnets.utils import calculate_output_dim


class TD3ActorNetwork(torch.nn.Module):
    def __init__(
        self, representation_network, actor_net, network_output_dim, action_shape
    ) -> None:
        super().__init__()

        self._action_shape = action_shape
        if actor_net is None:
            actor_network = torch.nn.Identity()
        else:
            actor_network = actor_net(network_output_dim)
        feature_dim = np.prod(calculate_output_dim(actor_network, network_output_dim))
        self.actor = torch.nn.Sequential(
            representation_network,
            actor_network,
            torch.nn.Flatten(),
            torch.nn.Linear(feature_dim, np.prod(action_shape)),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        x = self.actor(x)
        return torch.reshape(x, (x.size(0), *self._action_shape))


class TD3CriticNetwork(torch.nn.Module):
    def __init__(
        self,
        representation_network,
        critic_net,
        network_output_dim,
        n_critics,
        action_shape,
    ) -> None:
        super().__init__()
        self.network = representation_network
        if critic_net is None:
            critic_net = lambda x: torch.nn.Identity()
        self._n_critics = n_critics
        input_shape = (np.prod(network_output_dim) + np.prod(action_shape),)
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
        obs = self.network(obs)
        x = torch.cat([obs, actions], dim=1)
        return self._critics[0](x)
