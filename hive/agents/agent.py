import abc

from hive.utils.registry import Registrable


class Agent(abc.ABC, Registrable):
    """Base class for agents. Every implemented agent should be a subclass of
    this class.
    """

    def __init__(self, obs_dim, act_dim, id=0):
        """
        Args:
            obs_dim: Dimension of observations that agent will see.
            act_dim: Number of actions that the agent needs to chose from.
            id: Identifier for the agent.
        """
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._training = True
        self._id = str(id)

    @property
    def id(self):
        return self._id

    @abc.abstractmethod
    def act(self, observation):
        """Returns an action for the agent to perform based on the observation.

        Args:
            observation: Current observation that agent should act on.
        Returns:
            Action for the current timestep.
        """
        pass

    @abc.abstractmethod
    def update(self, update_info):
        """
        Updates the agent.

        Args:
            update_info (dict): Contains information agent needs to update
                itself.
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
