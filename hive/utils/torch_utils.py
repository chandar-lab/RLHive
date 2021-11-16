import numpy as np
import torch
from hive import registry
from hive.utils.utils import OptimizerFn
from torch import optim


def numpify(t):
    if isinstance(t, np.ndarray):
        return t
    elif isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    else:
        return np.array(t)


registry.register_all(
    OptimizerFn,
    {
        "Adadelta": OptimizerFn(optim.Adadelta),
        "Adagrad": OptimizerFn(optim.Adagrad),
        "Adam": OptimizerFn(optim.Adam),
        "Adamax": OptimizerFn(optim.Adamax),
        "AdamW": OptimizerFn(optim.AdamW),
        "ASGD": OptimizerFn(optim.ASGD),
        "LBFGS": OptimizerFn(optim.LBFGS),
        "RMSprop": OptimizerFn(optim.RMSprop),
        "Rprop": OptimizerFn(optim.Rprop),
        "SGD": OptimizerFn(optim.SGD),
        "SparseAdam": OptimizerFn(optim.SparseAdam),
    },
)

get_optimizer_fn = getattr(registry, f"get_{OptimizerFn.type_name()}")
