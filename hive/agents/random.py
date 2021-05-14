import os
import copy
import numpy as np
import torch
from hive.agents.agent import Agent


class RandomAgent(Agent):
    """A random agent"""

    def __init__(self, obs_dim, act_dim, id=0, seed=42, device="cpu", logger=None):
        """
        Args:
            obs_dim: The dimension of the observations.
            act_dim: The number of actions available to the agent.
            id: ID used to create the timescale in the logger for the agent.
            seed: Seed for numpy random number generator.
            device: Device on which all computations should be run.
            logger: Logger used to log agent's metrics.
        """
        super().__init__(obs_dim=obs_dim, act_dim=act_dim, id=f"random_agent_{id}")
        self._rng = np.random.default_rng(seed=seed)

    @torch.no_grad()
    def act(self, observation):
        """Returns a random action for the agent."""

        action = self._rng.integers(self._act_dim)

        return action

    def update(self, update_info):

        pass

    def save(self, dname):
        torch.save(
            {"rng": self._rng,}, os.path.join(dname, "agent.pt"),
        )

    def load(self, dname):
        checkpoint = torch.load(os.path.join(dname, "agent.pt"))
        self._rng = checkpoint["rng"]
