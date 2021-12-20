import abc

from hive.utils.registry import Registrable


class BaseReplayBuffer(abc.ABC, Registrable):
    """Base class for replay buffers. Every implemented buffer should be a subclass of this class."""

    @abc.abstractmethod
    def add(self, **data):
        """
        Adds data to the buffer

        Args:
            data: data to add to the replay buffer. Subclasses can define this class
                signature based on use case.
        """

    @abc.abstractmethod
    def sample(self, batch_size):
        """
        sample a minibatch

        Args:
            batch_size (int): the number of transitions to sample.
        """

    @abc.abstractmethod
    def size(self):
        """
        Returns the number of transitions stored in the buffer.
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
            dname (str): directory name where buffer checkpoint info is stored.

        Returns:
            True if successfully loaded the buffer. False otherwise.
        """

    @classmethod
    def type_name(cls):
        """
        Returns: "replay"
        """
        return "replay"
