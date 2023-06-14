import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
import torch

from hive.types import PathLike

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def create_folder(folder: PathLike):
    """Creates a folder.

    Args:
        folder (str): Folder to create.
    """
    Path(folder).mkdir(parents=True, exist_ok=True)


class Seeder:
    """Class used to manage seeding in RLHive. It sets the seed for all the frameworks
    that RLHive currently uses. It also deterministically provides new seeds based on
    the global seed, in case any other objects in RLHive (such as the agents) need
    their own seed.
    """

    def __init__(self):
        self._seed = 0
        self._current_seeds = defaultdict(lambda: self._seed)

    def set_global_seed(self, seed):
        """This reduces some sources of randomness in experiments. To get reproducible
        results, you must run on the same machine and set the environment variable
        CUBLAS_WORKSPACE_CONFIG to ":4096:8" or ":16:8" before starting the experiment.

        Args:
            seed (int): Global seed.
        """
        self._seed = seed
        self._current_seeds = defaultdict(lambda: self._seed)
        torch.manual_seed(self._seed)
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.backends.cudnn.benchmark = False  # type: ignore
        torch.use_deterministic_algorithms(True)

    def get_new_seed(self, group=None):
        """Each time it is called, it increments the current_seed for the
        requested group and returns it. If no group is specified, the default
        group is selected.

        Args:
            group (str): The name of the group to get the seed for.
        """
        seed = self._current_seeds[group]
        self._current_seeds[group] += 1
        return seed


seeder = Seeder()


def DL_to_LD(x, list_size):
    """Turns a dictionary of lists to a list of dictionaries."""
    return [{k: v[i] for k, v in x.items()} for i in range(list_size)]


def LD_to_DL(x):
    """Turns a list of dictionaries to a dictionary of lists."""
    return {k: [d[k] for d in x] for k in x[0]}


class Chomp(dict):
    """An extension of the dictionary class that allows for accessing through dot
    notation and easy saving/loading.
    """

    def __getattr__(self, k):
        if k not in self:
            raise AttributeError()
        return self.__getitem__(k)

    def __setattr__(self, k, v):
        self.__setitem__(k, v)

    def save(self, filename):
        """Saves the object using pickle.

        Args:
            filename (str): Filename to save object.
        """
        pickle.dump(self, open(filename, "wb"))

    def load(self, filename):
        """Loads the object.

        Args:
            filename (str): Where to load object from.
        """

        self.clear()
        self.update(pickle.load(open(filename, "rb")))


# class OptimizerFn(torch.optim.Optimizer):
#     """A wrapper for callables that produce optimizer functions.

#     These wrapped callables can be partially initialized through configuration
#     files or command line arguments.
#     """

#     @classmethod
#     def type_name(cls):
#         """
#         Returns:
#             "optimizer_fn"
#         """
#         return "optimizer_fn"


# class LossFn(Registrable):
#     """A wrapper for callables that produce loss functions.

#     These wrapped callables can be partially initialized through configuration
#     files or command line arguments.
#     """

#     @classmethod
#     def type_name(cls):
#         """
#         Returns:
#             "loss_fn"
#         """

#         return "loss_fn"
from typing import Any, NewType, Protocol, TypeVar, Union, runtime_checkable

T = TypeVar("T")

# LossFn = NewType("LossFn", torch.nn.Module)
# ActivationFn = NewType("ActivationFn", torch.nn.Module)

OptimizerFn = torch.optim.Optimizer


@runtime_checkable
class ActivationFn(Protocol):
    def __call__(self, x: T) -> T:
        ...


@runtime_checkable
class LossFn(Protocol):
    def __call__(self, pred, target) -> Any:
        ...


# LossFn = TypeVar("LossFn", bound=torch.nn.Module)
# LossFn = TypeVar("LossFn", bound=Callable)
# class ActivationFn(Registrable):
#     """A wrapper for callables that produce activation functions.

#     These wrapped callables can be partially initialized through configuration
#     files or command line arguments.
#     """


class Counter:
    """A wrapper for int that allows for incrementing and decrementing."""

    def __init__(self, value=0):
        self._value = value

    @property
    def value(self):
        """Returns the value of the counter."""
        return self._value

    def increment(self, steps=1):
        """Increments the counter."""
        self._value += steps

    def decrement(self, steps=1):
        """Decrements the counter."""
        self._value -= steps

    def __repr__(self):
        return f"Counter({self._value})"

    # Implement all operations used to emulate numeric type and comparison
    # operators.
    def __eq__(self, other):
        return self._value == other

    def __ne__(self, other):
        return self._value != other

    def __lt__(self, other):
        return self._value < other

    def __le__(self, other):
        return self._value <= other

    def __gt__(self, other):
        return self._value > other

    def __ge__(self, other):
        return self._value >= other

    def __add__(self, other):
        return self._value + other

    def __sub__(self, other):
        return self._value - other

    def __mul__(self, other):
        return self._value * other

    def __truediv__(self, other):
        return self._value / other

    def __floordiv__(self, other):
        return self._value // other

    def __mod__(self, other):
        return self._value % other

    def __divmod__(self, other):
        return divmod(self._value, other)

    def __pow__(self, other):
        return self._value**other

    def __radd__(self, other):
        return other + self._value

    def __rsub__(self, other):
        return other - self._value

    def __rmul__(self, other):
        return other * self._value

    def __rtruediv__(self, other):
        return other / self._value

    def __rfloordiv__(self, other):
        return other // self._value

    def __rmod__(self, other):
        return other % self._value

    def __rdivmod__(self, other):
        return divmod(other, self._value)

    def __rpow__(self, other):
        return other**self._value

    def __neg__(self):
        return -self._value

    def __pos__(self):
        return +self._value

    def __abs__(self):
        return abs(self._value)

    def __int__(self):
        return int(self._value)

    def __float__(self):
        return float(self._value)
