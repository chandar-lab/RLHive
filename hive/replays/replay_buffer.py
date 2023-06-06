import abc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Union, Tuple

import numpy as np


@dataclass(frozen=True)
class ReplayItemSpec:
    shape: Tuple[int, ...]
    dtype: Union[type, np.dtype]

    @classmethod
    def create(cls, shape: Sequence[int], dtype: Union[type, np.dtype, str]):
        return cls(tuple(shape), str_to_dtype(dtype))


def str_to_dtype(dtype: Union[type, np.dtype, str]) -> Union[type, np.dtype]:
    if isinstance(dtype, type) or isinstance(dtype, np.dtype):
        return dtype
    elif dtype.startswith("np.") or dtype.startswith("numpy."):
        return np.sctypeDict[dtype.split(".")[1]]
    else:
        type_dict = {
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
        }
        return type_dict[dtype]


class BaseReplayBuffer(abc.ABC):
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
    def add_transitions(self, **data):
        """
        Adds multiple transitions to the buffer.

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
