import abc

import gymnasium as gym

from hive.utils.registry import Registrable


class Agent(abc.ABC, Registrable):
    """Base class for agents. Every implemented agent should be a subclass of
    this class.
    """

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, id=0):
        """
        Args:
            observation_space (gym.Space): Observation space for agent.
            action_space (gym.Space): Action space for agent.
            id: Identifier for the agent.
        """
        self._observation_space = observation_space
        self._action_space = action_space
        self._training = True
        self._id = str(id)

    @property
    def id(self):
        return self._id

    @abc.abstractmethod
    def act(self, observation, agent_traj_state):
        """Returns an action for the agent to perform based on the observation.

        Args:
            observation: Current observation that agent should act on.
            agent_traj_state: Contains necessary state information for the agent
                to process current trajectory. This should be updated and returned.
        Returns:
            - Action for the current timestep.
            - Agent trajectory state.
        """
        pass

    @abc.abstractmethod
    def update(self, update_info):
        """
        Updates the agent.

        Args:
            update_info (dict): Contains information from the environment agent needs
                to update itself.
            agent_traj_state: Contains necessary state information for the agent
                to process current trajectory. This should be updated and returned.

        Returns:
            Agent trajectory state.
        """
        pass

    def train(self):
        """Changes the agent to training mode."""
        self._training = True

    def eval(self):
        """Changes the agent to evaluation mode"""
        self._training = False

    @abc.abstractmethod
    def save(self, dname):
        """
        Saves agent checkpointing information to file for future loading.

        Args:
            dname (str): directory where agent should save all relevant info.
        """
        pass

    @abc.abstractmethod
    def load(self, dname):
        """
        Loads agent information from file.

        Args:
            dname (str): directory where agent checkpoint info is stored.
        """
        pass

    @classmethod
    def type_name(cls):
        """
        Returns:
            "agent"
        """
        return "agent"
