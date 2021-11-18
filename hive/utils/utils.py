import os
import pickle
import random

import numpy as np
import torch

from hive import registry
from hive.utils.registry import CallableType


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


class Seeder:
    def __init__(self):
        self._seed = 0
        self._current_seed = 0

    def set_global_seed(self, seed):
        """This reduces some sources of randomness in experiments. To get reproducible
        results, you must run on the same machine and set the environment variable
        CUBLAS_WORKSPACE_CONFIG to ":4096:8" or ":16:8" before starting the experiment.
        """
        self._seed = seed
        self._current_seed = seed
        torch.manual_seed(self._seed)
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    def get_new_seed(self):
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
        pickle.dump(self, open(filename, "wb"))

    def load(self, filename):
        self.clear()
        self.update(pickle.load(open(filename, "rb")))


class OptimizerFn(CallableType):
    @classmethod
    def type_name(cls):
        return "optimizer_fn"
