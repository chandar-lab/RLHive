import math

import torch

from hive.utils.registry import registry
from hive.utils.utils import CallableType


def calculate_output_dim(net, input_shape):
    """Calculates the resulting output shape for a given input shape and network.

    Args:
        net (torch.nn.Module): The network which you want to calculate the output
            dimension for.
        input_shape (int | tuple[int]): The shape of the input being fed into the
            :obj:`net`. Batch dimension should not be included.
    Returns:
        The shape of the output of a network given an input shape.
        Batch dimension is not included.
    """
    if isinstance(input_shape, int):
        input_shape = (input_shape,)
    placeholder = torch.zeros((0,) + tuple(input_shape))
    output = net(placeholder)
    return output.size()[1:]


def create_init_weights_fn(initialization_fn):
    """Returns a function that wraps :func:`initialization_function` and applies
    it to modules that have the :attr:`weight` attribute.

    Args:
        initialization_fn (callable): A function that takes in a tensor and
            initializes it.
    Returns:
        Function that takes in PyTorch modules and initializes their weights.
        Can be used as follows:

        .. code-block:: python

            init_fn = create_init_weights_fn(variance_scaling_)
            network.apply(init_fn)
    """
    if initialization_fn is not None:

        def init_weights(m):
            if hasattr(m, "weight"):
                initialization_fn(m.weight)

        return init_weights
    else:
        return lambda m: None


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


class InitializationFn(CallableType):
    """A wrapper for callables that produce initialization functions.

    These wrapped callables can be partially initialized through configuration
    files or command line arguments.
    """

    @classmethod
    def type_name(cls):
        """
        Returns:
            "init_fn"
        """
        return "init_fn"


registry.register_all(
    InitializationFn,
    {
        "uniform": InitializationFn(torch.nn.init.uniform_),
        "normal": InitializationFn(torch.nn.init.normal_),
        "constant": InitializationFn(torch.nn.init.constant_),
        "ones": InitializationFn(torch.nn.init.ones_),
        "zeros": InitializationFn(torch.nn.init.zeros_),
        "eye": InitializationFn(torch.nn.init.eye_),
        "dirac": InitializationFn(torch.nn.init.dirac_),
        "xavier_uniform": InitializationFn(torch.nn.init.xavier_uniform_),
        "xavier_normal": InitializationFn(torch.nn.init.xavier_normal_),
        "kaiming_uniform": InitializationFn(torch.nn.init.kaiming_uniform_),
        "kaiming_normal": InitializationFn(torch.nn.init.kaiming_normal_),
        "orthogonal": InitializationFn(torch.nn.init.orthogonal_),
        "sparse": InitializationFn(torch.nn.init.sparse_),
        "variance_scaling": InitializationFn(variance_scaling_),
    },
)

get_optimizer_fn = getattr(registry, f"get_{InitializationFn.type_name()}")
