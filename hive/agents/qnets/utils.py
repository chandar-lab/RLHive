import torch
from hive.utils.utils import CallableType, registry


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
    },
)

get_optimizer_fn = getattr(registry, f"get_{InitializationFn.type_name()}")
