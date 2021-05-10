import abc


class Agent(abc.ABC):
    """Base class for agents. Every implemented agent should be a subclass of this class."""

    def __init__(self, obs_dim, act_dim, num_disc_per_obs_dim=None, id=0):
        """Constructor for Agent class.
        Args:
            obs_dim: dimension of observations that agent will see.
            act_dim: Number of actions that the agent needs to chose from.
            num_disc_per_obs_dim: Number of discrete observations per dimension
                of the observation space. Each dimension of the obs space can
                be represented as a one hot encoding with this parameter as
                the max value for the encoding. If None (default) then this
                means that observations should be treated as continuous inputs
            id: Identifier for the agent.
        """
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._num_disc_per_obs_dim = num_disc_per_obs_dim
        self._training = True
        self._id = str(id)

    @property
    def id(self):
        return self._id

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
