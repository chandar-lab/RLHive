import numpy as np
import torch

from hive.agents.qnets.utils import calculate_output_dim
from torch.distributions.categorical import Categorical

#Inspired from https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_shared.py and TD3 Agent 
class PPOActorCriticNetwork(torch.nn.Module):
    def __init__(
        self, representation_network, actor_net, critic_net, network_output_dim, action_space
    ) -> None:
        super().__init__()
        self.network = representation_network
        if actor_net is None:
            actor_network = torch.nn.Identity()
        else:
            actor_network = actor_net(network_output_dim)
        feature_dim = np.prod(calculate_output_dim(actor_network, network_output_dim))
        
        self.actor = torch.nn.Sequential(
            actor_network,
            torch.nn.Flatten(),
            torch.nn.Linear(feature_dim, action_space.n),
        )

        if critic_net is None:
            critic_network = torch.nn.Identity()
        else:
            critic_network  = critic_net(network_output_dim)
        feature_dim = np.prod(calculate_output_dim(critic_network, network_output_dim))
        self.critic = torch.nn.Sequential(
            critic_network,
            torch.nn.Flatten(),
            torch.nn.Linear(feature_dim, 1),
        )

    def forward(self, x):
        hidden = self.network(x)
        return self.actor(hidden)
    
    def get_value(self, x):
        hidden = self.network(x)
        return self.critic(hidden)
    
    def get_action_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
