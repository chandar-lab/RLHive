import pathlib
from functools import partial
from typing import Sequence, Tuple, Type, TypeVar, Union, cast

# from hive.utils.registry import PartialCreates

Shape = Sequence[int]
PathLike = Union[str, pathlib.Path]
T = TypeVar("T")
