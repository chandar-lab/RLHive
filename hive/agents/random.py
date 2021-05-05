import os
import copy
import numpy as np
import torch
from hive.agents.agent import Agent


class RandomAgent(Agent):
    """A random agent
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        id=0,
        seed=42,
        device="cpu",
        logger=None,
        log_frequency=100,
    ):
        """
        Args:
            obs_dim: The dimension of the observations.
            act_dim: The number of actions available to the agent.
            id: ID used to create the timescale in the logger for the agent.
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

        pass

    def load(self, dname):

        pass

