import math
from typing import (
    Any,
    Callable,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    runtime_checkable,
)

import optree
import torch

from hive.types import Partial
from hive.utils.registry import registry

T = TypeVar("T")


def calculate_output_dim(
    net: Callable[..., optree.PyTree[torch.Tensor]],
    input_shape: Union[int, Sequence[int]],
) -> Any:  # PyTree[Sequence[int]] Using Any to avoid checks for recursive types
    """Calculates the resulting output shape for a given input shape and network.

    Args:
        net (torch.nn.Module): The network which you want to calculate the output
            dimension for.
        input_shape (int | Sequence[int]): The shape of the input being fed into the
            :obj:`net`. Batch dimension should not be included.
    Returns:
        The shape of the output of a network given an input shape.
        Batch dimension is not included.
    """
    if isinstance(input_shape, int):
        input_shape = (input_shape,)
    placeholder = torch.zeros((1,) + tuple(input_shape))
    output = net(placeholder)

    def get_size(y: torch.Tensor) -> Sequence[int]:
        return y.size()[1:]

    return optree.tree_map(get_size, output)


def apply_to_tensor(
    x: optree.PyTree[torch.Tensor], fn: Callable[[torch.Tensor], T]
) -> optree.PyTree[T]:
    return optree.tree_map(fn, x)


def calculate_correct_fan(tensor, mode):
    """Calculate fan of tensor.

    Args:
        tensor (torch.Tensor): Tensor to calculate fan of.
        mode (str): Which type of fan to compute. Must be one of `"fan_in"`,
            `"fan_out"`, and `"fan_avg"`.
    Returns:
        Fan of the tensor based on the mode.
    """
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        return fan_in
    elif mode == "fan_out":
        return fan_out
    elif mode == "fan_avg":
        return (fan_in + fan_out) / 2
    else:
        raise ValueError(f"Fan mode {mode} not supported")


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="uniform"):
    """Implements the :py:class:`tf.keras.initializers.VarianceScaling`
    initializer in PyTorch.

    Args:
        tensor (torch.Tensor): Tensor to initialize.
        scale (float): Scaling factor (must be positive).
        mode (str): Must be one of `"fan_in"`, `"fan_out"`, and `"fan_avg"`.
        distribution: Random distribution to use, must be one of
            "truncated_normal", "untruncated_normal" and "uniform".
    Returns:
        Initialized tensor.
    """
    fan = calculate_correct_fan(tensor, mode)
    scale /= fan
    if distribution == "truncated_normal":
        # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        stddev = math.sqrt(scale) / 0.87962566103423978
        return torch.nn.init.trunc_normal_(tensor, 0.0, stddev, -2 * stddev, 2 * stddev)
    elif distribution == "untruncated_normal":
        stddev = math.sqrt(scale)
        return torch.nn.init.normal_(tensor, 0.0, stddev)
    elif distribution == "uniform":
        limit = math.sqrt(3.0 * scale)
        return torch.nn.init.uniform_(tensor, -limit, limit)
    else:
        raise ValueError(f"Distribution {distribution} not supported")


@runtime_checkable
class TensorInitFn(Protocol):
    def __call__(self, tensor: torch.Tensor):
        ...


@runtime_checkable
class ModuleInitFn(Protocol):
    def __call__(self, module: torch.nn.Module):
        ...


def layer_init(
    weight_init_fn: Optional[Partial[TensorInitFn]],
    bias_init_fn: Optional[Partial[TensorInitFn]],
) -> ModuleInitFn:
    def init_fn(module: torch.nn.Module):
        if hasattr(module, "weight") and weight_init_fn is not None:
            weight_init_fn(module.weight)  # type: ignore
        if hasattr(module, "bias") and bias_init_fn is not None:
            bias_init_fn(module.bias)  # type: ignore

    return init_fn


registry.register_all_with_type(
    TensorInitFn,
    {
        "uniform": torch.nn.init.uniform_,
        "normal": torch.nn.init.normal_,
        "constant": torch.nn.init.constant_,
        "ones": torch.nn.init.ones_,
        "zeros": torch.nn.init.zeros_,
        "eye": torch.nn.init.eye_,
        "dirac": torch.nn.init.dirac_,
        "xavier_uniform": torch.nn.init.xavier_uniform_,
        "xavier_normal": torch.nn.init.xavier_normal_,
        "kaiming_uniform": torch.nn.init.kaiming_uniform_,
        "kaiming_normal": torch.nn.init.kaiming_normal_,
        "orthogonal": torch.nn.init.orthogonal_,
        "sparse": torch.nn.init.sparse_,
        "variance_scaling": variance_scaling_,
    },
)
