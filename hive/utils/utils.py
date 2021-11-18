import os
import pickle

from hive.utils.registry import CallableType


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


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
