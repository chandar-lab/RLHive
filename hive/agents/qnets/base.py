from abc import ABC
from hive.utils.registry import CallableType


class FunctionApproximator(CallableType):
    @classmethod
    def type_name(cls):
        return "function"
