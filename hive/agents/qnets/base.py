from abc import ABC
from hive import Registrable


class FunctionApproximator(ABC, Registrable):
    def __call__(self, **kwargs):
        raise NotImplementedError

    @classmethod
    def type_name(cls):
        return "function"
