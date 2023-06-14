import pathlib
from typing import Callable, Optional, Sequence, TypeVar, Union

from typing_extensions import Annotated

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
C = TypeVar("C", bound=Callable)
Creates = Annotated[Callable[..., T_co], "configured", "creates"]
Partial = Annotated[C, "configured", "partial"]
PathLike = Union[str, pathlib.Path]


def default(fn: Optional[T], default_fn: T) -> T:
    if fn is None:
        return default_fn
    else:
        return fn
