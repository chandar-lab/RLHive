import torch
from hive.utils.utils import CallableType, registry
import math


def calculate_output_dim(net, input_dim):
    if isinstance(input_dim, int):
        input_dim = tuple(input_dim)
    placeholder = torch.zeros((0,) + tuple(input_dim))
    output = net(placeholder)
    return output.size()[1:]


def create_init_weights_fn(initialization_fn):
    if initialization_fn is not None:

        def init_weights(m):
            if hasattr(m, "weight"):
                initialization_fn(m.weight)

        return init_weights
    else:
        return lambda m: None


def calculate_correct_fan(tensor, mode):
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
    fan = calculate_correct_fan(tensor, mode)
    scale /= fan
    if distribution == "truncated_normal":
        # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        stddev = math.sqrt(scale) / 0.87962566103423978
        return torch.nn.init.trunc_normal_(tensor, 0.0, stddev, -2 * stddev, 2 * stddev)
    elif distribution == "untruncated_normal":
        stddev = math.sqrt(scale)
        return torch.nn.init.normal_(tensor, 0.0, stddev)
    else:
        limit = math.sqrt(3.0 * scale)
        return torch.nn.init.uniform_(tensor, -limit, limit)


class InitializationFn(CallableType):
    @classmethod
    def type_name(cls):
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
