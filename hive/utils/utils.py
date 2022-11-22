from collections import defaultdict
import os
import pickle
import random

import numpy as np
import torch
from numpy.random._generator import Generator

from hive.utils.registry import Registrable

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MyGenerator(Generator):
    def randint(
        self: Generator,
        low: int,
        high: int,
        size=None,
        dtype="l",
        endpoint: bool = False,
    ):
        """Replacement for `numpy.random.Generator.randint` that uses the `Generator.integers` method
        instead of `Generator.random_integers` which is deprecated."""
        return self.integers(low, high, size=size, dtype=dtype, endpoint=endpoint)


def _patched_np_random(seed: int = None) -> tuple[MyGenerator, int]:
    """Replacement for `gym.utils.seeding.np_random` that uses the `MyGenerator` class instead of
    `numpy.random.Generator`. MyGenerator has a `.randint` method so the old code from marlenv
    can still work."""
    from gym import error

    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        raise error.Error(f"Seed must be a non-negative integer or omitted, not {seed}")
    seed_seq = np.random.SeedSequence(seed)
    np_seed = seed_seq.entropy
    rng = MyGenerator(np.random.PCG64(seed_seq))
    return rng, np_seed


def create_folder(folder):
    """Creates a folder.

    Args:
        folder (str): Folder to create.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)


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
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    def get_new_seed(self, group=None):
        """Each time it is called, it increments the current_seed for the
        requested group and returns it. If no group is specified, the default
        group is selected.

        Args:
            group (str): The name of the group to get the seed for.
        """
        self._current_seeds[group] += 1
        return self._current_seeds[group]


seeder = Seeder()


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


class OptimizerFn(Registrable):
    """A wrapper for callables that produce optimizer functions.

    These wrapped callables can be partially initialized through configuration
    files or command line arguments.
    """

    @classmethod
    def type_name(cls):
        """
        Returns:
            "optimizer_fn"
        """
        return "optimizer_fn"


class LossFn(Registrable):
    """A wrapper for callables that produce loss functions.

    These wrapped callables can be partially initialized through configuration
    files or command line arguments.
    """

    @classmethod
    def type_name(cls):
        """
        Returns:
            "loss_fn"
        """
        return "loss_fn"


class ActivationFn(Registrable):
    """A wrapper for callables that produce activation functions.

    These wrapped callables can be partially initialized through configuration
    files or command line arguments.
    """

    @classmethod
    def type_name(cls):
        """
        Returns:
            "activation_fn"
        """
        return "activation_fn"
