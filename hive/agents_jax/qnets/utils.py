import math

import jax
import jax.numpy as jnp

from hive.utils.registry import registry
from hive.utils.utils import CallableType


def calculate_output_dim(net, input_shape):
    """Calculates the resulting output shape for a given input shape and network.
    Args:
        net (flax.nn.Module): The network which you want to calculate the output
            dimension for.
        input_shape (int | tuple[int]): The shape of the input being fed into the
            :obj:`net`. Batch dimension should not be included.
    Returns:
        The shape of the output of a network given an input shape.
        Batch dimension is not included.
    """
    if isinstance(input_shape, int):
        input_shape = (input_shape,)
    placeholder = jnp.zeros((0,) + tuple(input_shape))
    _rng = jax.random.PRNGKey(1)
    _, rng = jax.random.split(_rng)
    params = net.init(rng)
    output = net.apply(params, placeholder)  ## does not work
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
        "uniform": InitializationFn(jax.nn.initializers.uniform),
        "normal": InitializationFn(jax.nn.initializers.normal),
        "constant": InitializationFn(jax.nn.initializers.constant),
        "ones": InitializationFn(jax.nn.initializers.ones),
        "zeros": InitializationFn(jax.nn.initializers.zeros),
        # "eye": InitializationFn(torch.nn.init.eye_), ## Not in Jax
        # "dirac": InitializationFn(torch.nn.init.dirac_), ## Not in Jax
        "xavier_uniform": InitializationFn(jax.nn.initializers.glorot_uniform),
        "xavier_normal": InitializationFn(jax.nn.initializers.glorot_normal),
        "kaiming_uniform": InitializationFn(jax.nn.initializers.he_uniform),
        "kaiming_normal": InitializationFn(jax.nn.initializers.he_normal),
        "orthogonal": InitializationFn(jax.nn.initializers.orthogonal),
        # "sparse": InitializationFn(torch.nn.init.sparse_), ## Not in Jax
        "variance_scaling": InitializationFn(jax.nn.initializers.variance_scaling),
        "lecun_uniform": InitializationFn(jax.nn.initializers.lecun_uniform),
        "lecun_normal": InitializationFn(jax.nn.initializers.lecun_normal),
    },
)

get_optimizer_fn = getattr(registry, f"get_{InitializationFn.type_name()}")
