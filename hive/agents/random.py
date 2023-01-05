import os

import gymnasium as gym
import numpy as np
import torch

from hive.agents.agent import Agent
from hive.utils.utils import seeder


class RandomAgent(Agent):
    """An agent that takes random steps at each timestep."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        id=0,
        logger=None,
        **kwargs
    ):
        """
        Args:
            observation_space (gym.Space): The shape of the observations.
            action_space (gym.Space): The number of actions available to the agent.
            id: Agent identifier.
            logger (ScheduledLogger): Logger used to log agent's metrics.
        """
        super().__init__(
            observation_space=observation_space, action_space=action_space, id=id
        )
        self._action_space.seed(seed=seeder.get_new_seed("agent"))

    @torch.no_grad()
    def act(self, observation, state=None):
        """Returns a random action for the agent."""
        action = self._action_space.sample()
        return action, state

    def update(self, update_info, state=None):
        return state

    def save(self, dname):
        torch.save(
            {"action_space": self._action_space},
            os.path.join(dname, "agent.pt"),
        )

    def load(self, dname):
        checkpoint = torch.load(os.path.join(dname, "agent.pt"))
        self._action_space = checkpoint["action_space"]
