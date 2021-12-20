import os

import numpy as np
import torch

from hive.agents.agent import Agent
from hive.utils.utils import seeder


class RandomAgent(Agent):
    """An agent that takes random steps at each timestep."""

    def __init__(self, obs_dim, act_dim, id=0, logger=None):
        """
        Args:
            obs_dim: The shape of the observations.
            act_dim (int): The number of actions available to the agent.
            id: Agent identifier.
            logger (ScheduledLogger): Logger used to log agent's metrics.
        """
        super().__init__(obs_dim=obs_dim, act_dim=act_dim, id=id)
        self._rng = np.random.default_rng(seed=seeder.get_new_seed())

    @torch.no_grad()
    def act(self, observation):
        """Returns a random action for the agent."""

        action = self._rng.integers(self._act_dim)

        return action

    def update(self, update_info):

        pass

    def save(self, dname):
        torch.save(
            {
                "rng": self._rng,
            },
            os.path.join(dname, "agent.pt"),
        )

    def load(self, dname):
        checkpoint = torch.load(os.path.join(dname, "agent.pt"))
        self._rng = checkpoint["rng"]
