import abc


class Agent(abc.ABC):
    """Base class for agents. Every implemented agent should be a subclass of this class."""

    @abc.abstractmethod
    def act(self, observation):
        """Returns an action for the agent to perform based on the observation"""

    @abc.abstractmethod
    def update(self, update_info):
        """
        Updates the agent. 
        
        Args:
            update_info: dictionary containing information agent needs to update itself.
        """

    @abc.abstractmethod
    def save(self, dname):
        """
        Saves agent checkpointing information to file for future loading.
        
        Args:
            dname: directory where agent should save all relevant info.
        """

    @abc.abstractmethod
    def load(self, dname):
        """
        Loads agent information from file.
        
        Args:
            dname: directory where agent checkpoint info is stored.

        Returns:
            True if successfully loaded agent. False otherwise.
        """
