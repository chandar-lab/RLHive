import os
import pickle
import random

import numpy as np
import torch

from hive.utils.registry import CallableType

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
        self._current_seed = 0

    def set_global_seed(self, seed):
        """This reduces some sources of randomness in experiments. To get reproducible
        results, you must run on the same machine and set the environment variable
        CUBLAS_WORKSPACE_CONFIG to ":4096:8" or ":16:8" before starting the experiment.

        Args:
            seed (int): Global seed.
        """
        self._seed = seed
        self._current_seed = seed
        torch.manual_seed(self._seed)
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    def get_new_seed(self):
        """Each time it is called, it increments the current_seed and returns it."""
        self._current_seed += 1
        return self._current_seed


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


class OptimizerFn(CallableType):
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


class LossFn(CallableType):
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
