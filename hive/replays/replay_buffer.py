import abc
import os
import pickle

import numpy as np
from hive import Registrable
from hive.utils.utils import create_folder


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
        return "replay"


class CircularReplayBuffer(BaseReplayBuffer):
    """A simple circular replay buffers.

    Args:
            capacity (int): repaly buffer capacity
            compress (bool): if False, convert data to float32 otherwise keep it as int8.
            seed (int): Seed for a pseudo-random number generator.
    """

    def __init__(self, capacity=1e5, compress=False, seed=42, **kwargs):

        self._numpy_rng = np.random.default_rng(seed)
        self._capacity = int(capacity)
        self._compress = compress

        self._dtype = {
            "observation": "int8" if self._compress else "float32",
            "action": "int8",
            "reward": "int8" if self._compress else "float32",
            "next_observation": "int8" if self._compress else "float32",
            "done": "int8" if self._compress else "float32",
        }

        self._data = {}
        for data_key in self._dtype:
            self._data[data_key] = [None] * int(capacity)

        self._write_index = -1
        self._n = 0
        self._previous_transition = None

    def add(self, observation, action, reward, done, **kwargs):
        """
        Adds transition to the buffer

        Args:
            observation: The current observation
            action: The action taken on the current observation
            reward: The reward from taking action at current observation
            done: If current observation was the last observation in the episode
        """
        if self._previous_transition is not None:
            self._previous_transition["next_observation"] = observation
            self._write_index = (self._write_index + 1) % self._capacity
            self._n = int(min(self._capacity, self._n + 1))
            for key in self._data:
                self._data[key][self._write_index] = np.asarray(
                    self._previous_transition[key], dtype=self._dtype[key]
                )
        self._previous_transition = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "done": done,
        }

    def sample(self, batch_size=32):
        """
        sample a minibatch

        Args:
            batch_size (int): The number of examples to sample.
        """
        if self.size() == 0:
            raise ValueError("Buffer does not have any transitions yet." % batch_size)

        indices = self._numpy_rng.integers(self._n, size=batch_size)
        rval = {}
        for key in self._data:
            rval[key] = np.asarray(
                [self._data[key][idx] for idx in indices], dtype="float32"
            )

        return rval

    def size(self):
        """
        returns the number of transitions stored in the replay buffer
        """
        return self._n

    def save(self, dname):
        """
        Saves buffer checkpointing information to file for future loading.

        Args:
            dname (str): directory name where agent should save all relevant info.
        """
        create_folder(dname)

        sdict = {}
        sdict["capacity"] = self._capacity
        sdict["write_index"] = self._write_index
        sdict["n"] = self._n
        sdict["data"] = self._data

        full_name = os.path.join(dname, "meta.ckpt")
        sdict_data = pickle.dumps(sdict)
        max_bytes = 2 ** 31 - 1
        with open(full_name, "wb") as f:
            for idx in range(0, len(sdict_data), max_bytes):
                f.write(sdict_data[idx:idx + max_bytes])

    def load(self, dname):
        """
        Loads buffer from file.

        Args:
            dname (str): directory name where buffer checkpoint info is stored.

        Returns:
            True if successfully loaded the buffer. False otherwise.
        """
        full_name = os.path.join(dname, "meta.ckpt")
        sdict_data = bytearray(0)
        sdict_data_size = os.path.getsize(full_name)
        max_bytes = 2 ** 31 - 1
        with open(full_name, "rb") as f:
            for _ in range(0, sdict_data_size, max_bytes):
                sdict_data += f.read(max_bytes)
        sdict = pickle.loads(sdict_data)

        self._capacity = sdict["capacity"]
        self._write_index = sdict["write_index"]
        self._n = sdict["n"]
        self._data = sdict["data"]
