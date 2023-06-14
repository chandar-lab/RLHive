import abc
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Sequence, Tuple, Union

import numpy as np


class Alignment(IntEnum):
    """The alignment of the item to retrieve relative to the stack of observations.

    When the state is composed of a stack of observations, the item to retrieve from the
    replay buffer can be aligned with the start or the end of the stack. For
    example, in the case of observations, the items to retrieve are necessarily
    aligned with the start of the stack. When retrieving actions, you want to
    retrieve actions corresponding to the end of the stack of observations.

    In the following example, the stack size is 4:
      start retrieving from here -> O1 O2 O3 O4
                                    A1 A2 A3 A4 <- retrieve from here

    """

    end = 0  # The item to retrieve is aligned with the end of the observation
    start = 1  # The item to retrieve is aligned with the start of the observation


@dataclass(frozen=True)
class ReplayItemSpec:
    shape: Tuple[int, ...]
    dtype: Union[type, np.dtype]
    return_next: bool = False
    alignment: Alignment = Alignment.end
    num_to_retrieve: int = 1

    @classmethod
    def create(
        cls,
        shape: Optional[Sequence[int]],
        dtype: Optional[Union[type, np.dtype, str]],
        return_next: bool = False,
        alignment: Alignment = Alignment.end,
        num_to_retrieve: int = 1,
    ):
        if shape is None:
            shape = ()
        if dtype is None:
            dtype = np.float32
        return cls(
            tuple(shape), str_to_dtype(dtype), return_next, alignment, num_to_retrieve
        )


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
    """Base class for replay buffers. Every implemented buffer should be a
    subclass of this class."""

    @abc.abstractmethod
    def add(self, **data):
        """
        Adds data to the buffer

        Args:
            data: data to add to the replay buffer. Subclasses can define this class
                signature based on use case.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def add_transitions(self, **data):
        """
        Adds multiple transitions to the buffer.

        Args:
            data: data to add to the replay buffer. Subclasses can define this class
                signature based on use case.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, batch_size):
        """
        sample a minibatch

        Args:
            batch_size (int): the number of transitions to sample.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def size(self):
        """
        Returns the number of transitions stored in the buffer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, dname):
        """
        Saves buffer checkpointing information to file for future loading.

        Args:
            dname (str): directory where agent should save all relevant info.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, dname):
        """
        Loads buffer from file.

        Args:
            dname (str): directory name where buffer checkpoint info is stored.

        Returns:
            True if successfully loaded the buffer. False otherwise.
        """
        raise NotImplementedError
