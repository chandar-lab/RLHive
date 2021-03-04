import abc


class Agent(abc.ABC):
    """Base class for agents. Every implemented agent should be a subclass of this class."""

    def __init__(self):
        self._training = True

    @abc.abstractmethod
    def act(self, observation):
        """Returns an action for the agent to perform based on the observation"""
        pass

    @abc.abstractmethod
    def update(self, update_info):
        """
        Updates the agent. 
        
        Args:
            update_info: dictionary containing information agent needs to update itself.
        """
        pass

    def train():
        """Changes the agent to training mode."""
        self._training = True

    def eval():
        """Changes the agent to evaluation mode"""
        self._training = False

    @abc.abstractmethod
    def save(self, dname):
        """
        Saves agent checkpointing information to file for future loading.
        
        Args:
            dname: directory where agent should save all relevant info.
        """
        pass

    @abc.abstractmethod
    def load(self, dname):
        """
        Loads agent information from file.
        
        Args:
            dname: directory where agent checkpoint info is stored.

        Returns:
            True if successfully loaded agent. False otherwise.
        """
        pass
