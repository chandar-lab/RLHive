import numpy as np
import torch

from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.utils import calculate_output_dim

class PPOCriticNetwork(torch.nn.Module):
    def __init__(
        self,
        representation_network: torch.nn.Module,
        critic_net: FunctionApproximator,
        network_output_shape
    ) -> None:
        """
        Args:
            representation_network (torch.nn.Module): Network that encodes the
                observations.
            critic_net (FunctionApproximator): Function that takes in the shape of the
                encoded observations and creates a network. This network takes one
                inputs: the encoded observations from representation_net.
                It outputs the representations used to compute the values of the
                states (ie everything except the last layer).
            network_output_shape: Expected output shape of the representation network.
        """
        super().__init__()
        self.network = representation_network
        if critic_net is None:
            critic_net = lambda x: torch.nn.Identity()
        input_shape = (np.prod(network_output_shape), )
        critic = critic_net(input_shape)
        feature_dim = np.prod(calculate_output_dim(critic, input_shape=input_shape))
        self._critics = torch.nn.Sequential(
                    critic,
                    torch.nn.Flatten(),
                    torch.nn.Linear(feature_dim, 1),
                )

    def forward(self, obs):
        obs = self.network(obs)
        obs = torch.flatten(obs, start_dim=1)
        return self._critic(obs)

#TODO: add Multi-Discrete
class CategoricalHead(torch.nn.Module):
    def __init__(self, feature_dim, action_space) -> None:
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
        self.network = torch.nn.Linear(feature_dim, action_space.n)
        self.distribution = torch.distributions.categorical.Categorical

    def forward(self, x):
        logits = self.network(x)
        return self.distribution(logits=logits)

class GaussianPolicyHead(torch.nn.Module):
    def __init__(self, feature_dim, action_space):
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
        self._action_shape = action_space.shape
        self.policy_mean = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, np.prod(self._action_shape)),
            torch.nn.Tanh()
        )
        self.policy_logstd = torch.nn.Parameter(torch.zeros(1, np.prod(action_space.shape)))
        self.distribution = torch.distributions.normal.Normal

    def forward(self, x):
        distribution = self.distribution(
            self.policy_mean(x),
            self.policy_logstd.repeat(x.shape[0], 1).exp()
        )
        return distribution

class PPOActorNetwork(torch.nn.Module):
    """A module that implements the TD3 actor computation. It puts together the
    :obj:`representation_network` and :obj:`actor_net`, and adds a final
    :py:class:`~torch.nn.Linear` layer to compute the action."""
    def __init__(
        self, representation_network, actor_net, network_output_dim, action_space, continuous_action
    ) -> None:
        super().__init__()
        self._continuous_action = continuous_action
        self.network = representation_network
        if actor_net is None:
            actor_network = torch.nn.Identity()
        else:
            actor_network = actor_net(network_output_dim)
        feature_dim = np.prod(calculate_output_dim(actor_network, network_output_dim))
        actor_head = GaussianPolicyHead if self._continuous_action else CategoricalHead

        self.actor = torch.nn.Sequential(
            actor_network,
            torch.nn.Flatten(),
            actor_head(feature_dim, action_space)
        )
    
    def forward(self, x, action=None):
        hidden = self.network(x)
        distribution = self.actor(hidden)
        if action is None:
            action = distribution.sample()
        
        logprob, entropy = distribution.log_prob(action), distribution.entropy()
        if self._continuous_action:
            logprob, entropy = logprob.sum(-1), entropy.mean(-1)
        return action, logprob, entropy