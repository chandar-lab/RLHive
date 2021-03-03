import os
import abc
import numpy as np
import _pickle as pickle

from hive.utils.utils import create_folder


class BaseReplayBuffer(abc.ABC):
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
    def save(self, fname):
        """
        Saves buffer checkpointing information to file for future loading.

        Args:
            fname (str): directory and file name where agent should save all relevant info.
        """

    @abc.abstractmethod
    def load(self, fname):
        """
        Loads buffer from file.

        Args:
            fname (str): directory and file name where buffer checkpoint info is stored.

        Returns:
            True if successfully loaded the buffer. False otherwise.
        """


class CircularReplayBuffer(BaseReplayBuffer):
    """A simple circular replay buffers.

    Args:
            numpy_rng (int): Container for a pseudo-random number generator.
            size (int): repaly buffer capacity
            compress (bool): if False, convert data to float32 otherwise keep it as int8.
    """

    def __init__(self, numpy_rng, size=1e5, compress=False):

        self._numpy_rng = numpy_rng
        self._size = int(size)
        self._compress = compress

        self._dtype = {
            "observations": "int8" if self._compress else "float32",
            "actions": "int8",
            "rewards": "int8" if self._compress else "float32",
            "next_observations": "int8" if self._compress else "float32",
            "done": "int8" if self._compress else "float32",
        }

        self._data = {}
        for data_key in self._dtype:
            self._data[data_key] = [None] * int(size)

        self._write_index = -1
        self._n = 0

    def add(self, data):
        """
        Adds data to the buffer

        Args:
            data (tuple): (state, action, reward, next_state)
        """
        self._write_index = (self._write_index + 1) % self._size
        self._n = int(min(self._size, self._n + 1))
        for idx, key in enumerate(self._data):
            self._data[key][self._write_index] = np.asarray(
                data[idx], dtype=self._dtype[key]
            )

    def sample(self, batch_size=32):
        """
        sample a minibatch

        Args:
            batch_size (int): .
        """
        if self._n < batch_size:
            raise IndexError(
                "Buffer does not have batch_size=%d transitions yet." % batch_size
            )

        indices = self._numpy_rng.choice(self._n, size=batch_size, replace=False)
        rval = {}
        for key in self._data:
            rval[key] = np.asarray(
                [self._data[key][idx] for idx in indices], dtype="float32"
            )

        return rval

    def size(self):
        """
        returns replay buffer size
        """
        return self._n

    def save(self, fname):
        """
        Saves buffer checkpointing information to file for future loading.

        Args:
            fname (str): directory and file name where agent should save all relevant info.
        """
        create_folder(fname)

        sdict = {}
        sdict["size"] = self._size
        sdict["write_index"] = self._write_index
        sdict["n"] = self._n

        full_name = os.path.join(fname, "meta.ckpt")
        with open(full_name, "wb") as f:
            pickle.dump(sdict, f)

        for key in self._data:
            full_name = os.path.join(fname, "{}.npy".format(key))
            with open(full_name, "w") as f:
                np.save(f, self._data[key])

    def load(self, fname):
        """
        Loads buffer from file.

        Args:
            fname (str): directory and file name where buffer checkpoint info is stored.

        Returns:
            True if successfully loaded the buffer. False otherwise.
        """
        full_name = os.path.join(fname, "meta.ckpt")
        with open(full_name, "rb") as f:
            sdict = pickle.load(f)

        self._size = sdict["size"]
        self._write_index = sdict["write_index"]
        self._n = sdict["n"]

        for key in self._data:
            full_name = os.path.join(fname, "{}.npy".format(key))
            with open(full_name, "r") as f:
                self._data[key] = np.load(f)
