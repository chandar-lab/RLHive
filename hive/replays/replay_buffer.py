import abc

class ReplayBuffer(abc.ABC):
    """Base class for replay buffers. Every implemented buffer should be a subclass of this class."""

    @abc.abstractmethod
    def add(self, data):
        """
        Adds data to the buffer

        Args:
            data (tuple): (state, action, reward, next_state)
        """

    @abc.abstractmethod
    def sample(self, batch_size):
        """
        sample a minibatch

        Args:
            batch_size (int): .
        """

    @abc.abstractmethod
    def size(self):
        """
        returns replay buffer size
        """

    @abc.abstractmethod
    def save(self, dname):
        """
        Saves buffer checkpointing information to file for future loading.

        Args:
            dname (str): directory where agent should save all relevant info.
        """

    @abc.abstractmethod
    def load(self, dname):
        """
        Loads buffer from file.

        Args:
            dname (str): directory where buffer checkpoint info is stored.

        Returns:
            True if successfully loaded the buffer. False otherwise.
        """
