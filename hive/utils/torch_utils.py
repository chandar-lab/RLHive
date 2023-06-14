import inspect
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

import numpy as np
import optree
import torch
from torch import optim

from hive.types import Partial
from hive.utils.registry import registry
from hive.utils.utils import ActivationFn, LossFn

T = TypeVar("T")


@runtime_checkable
class ModuleInitFn(Protocol):
    def __call__(self, module: torch.nn.Module) -> None:
        ...


@runtime_checkable
class TensorInitFn(Protocol):
    def __call__(self, tensor: torch.Tensor) -> None:
        ...


class RMSpropTF(optim.Optimizer):
    """
    Direct cut-paste from rwhightman/pytorch-image-models.
    https://github.com/rwightman/pytorch-image-models/blob/f7d210d759beb00a3d0834a3ce2d93f6e17f3d38/timm/optim/rmsprop_tf.py
    Licensed under Apache 2.0, https://github.com/rwightman/pytorch-image-models/blob/master/LICENSE

    Implements RMSprop algorithm (TensorFlow style epsilon)

    NOTE: This is a direct cut-and-paste of PyTorch RMSprop with eps applied before sqrt
    and a few other modifications to closer match Tensorflow for matching hyper-params.
    Noteworthy changes include:

    1. Epsilon applied inside square-root
    2. square_avg initialized to ones
    3. LR scaling of update accumulated in momentum buffer

    Proposed by G. Hinton in his
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.
    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing (decay) constant (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        decoupled_decay (bool, optional): decoupled weight decay as per https://arxiv.org/abs/1711.05101
        lr_in_momentum (bool, optional): learning rate scaling is included in the momentum buffer
            update as per defaults in Tensorflow

    """

    def __init__(
        self,
        params,
        lr=1e-2,
        alpha=0.9,
        eps=1e-10,
        weight_decay=0,
        momentum=0.0,
        centered=False,
        decoupled_decay=False,
        lr_in_momentum=True,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            alpha=alpha,
            eps=eps,
            centered=centered,
            weight_decay=weight_decay,
            decoupled_decay=decoupled_decay,
            lr_in_momentum=lr_in_momentum,
        )
        super(RMSpropTF, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSpropTF, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("momentum", 0)
            group.setdefault("centered", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("RMSprop does not support sparse gradients")
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["square_avg"] = torch.ones_like(p)  # PyTorch inits to zero
                    if group["momentum"] > 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    if group["centered"]:
                        state["grad_avg"] = torch.zeros_like(p)

                square_avg = state["square_avg"]
                one_minus_alpha = 1.0 - group["alpha"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    if group["decoupled_decay"]:
                        p.mul_(1.0 - group["lr"] * group["weight_decay"])
                    else:
                        grad = grad.add(p, alpha=group["weight_decay"])

                # Tensorflow order of ops for updating squared avg
                square_avg.add_(grad.pow(2) - square_avg, alpha=one_minus_alpha)
                # square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)  # PyTorch original

                if group["centered"]:
                    grad_avg = state["grad_avg"]
                    grad_avg.add_(grad - grad_avg, alpha=one_minus_alpha)
                    avg = (
                        square_avg.addcmul(grad_avg, grad_avg, value=-1)
                        .add(group["eps"])
                        .sqrt_()
                    )  # eps in sqrt
                    # grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)  # PyTorch original
                else:
                    avg = square_avg.add(group["eps"]).sqrt_()  # eps moved in sqrt

                if group["momentum"] > 0:
                    buf = state["momentum_buffer"]
                    # Tensorflow accumulates the LR scaling in the momentum buffer
                    if group["lr_in_momentum"]:
                        buf.mul_(group["momentum"]).addcdiv_(
                            grad, avg, value=group["lr"]
                        )
                        p.add_(-buf)
                    else:
                        # PyTorch scales the param update by LR
                        buf.mul_(group["momentum"]).addcdiv_(grad, avg)
                        p.add_(buf, alpha=-group["lr"])
                else:
                    p.addcdiv_(grad, avg, value=-group["lr"])

        return loss


def numpify(t):
    """Convert object to a numpy array.

    Args:
        t (np.ndarray | torch.Tensor | obj): Converts object to :py:class:`np.ndarray`.
    """
    if isinstance(t, np.ndarray):
        return t
    elif isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    else:
        return np.array(t)


def layer_init(
    module: torch.nn.Module,
    weight_init_fn: Optional[Partial[TensorInitFn]] = None,
    bias_init_fn: Optional[Partial[TensorInitFn]] = None,
):
    if hasattr(module, "weight") and weight_init_fn is not None:
        weight_init_fn(module.weight)  # type: ignore
    if hasattr(module, "bias") and bias_init_fn is not None:
        bias_init_fn(module.bias)  # type: ignore


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


registry.register("layer_init", layer_init, ModuleInitFn)

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
registry.register_classes(
    {
        "Adadelta": optim.Adadelta,
        "Adagrad": optim.Adagrad,
        "Adam": optim.Adam,
        "Adamax": optim.Adamax,
        "AdamW": optim.AdamW,
        "ASGD": optim.ASGD,
        "LBFGS": optim.LBFGS,
        "RMSprop": optim.RMSprop,
        "RMSpropTF": RMSpropTF,
        "Rprop": optim.Rprop,
        "SGD": optim.SGD,
        "SparseAdam": optim.SparseAdam,
    },
)

registry.register_all_with_type(
    LossFn,
    {
        "BCELoss": torch.nn.BCELoss,
        "BCEWithLogitsLoss": torch.nn.BCEWithLogitsLoss,
        "CosineEmbeddingLoss": torch.nn.CosineEmbeddingLoss,
        "CrossEntropyLoss": torch.nn.CrossEntropyLoss,
        "CTCLoss": torch.nn.CTCLoss,
        "HingeEmbeddingLoss": torch.nn.HingeEmbeddingLoss,
        "KLDivLoss": torch.nn.KLDivLoss,
        "L1Loss": torch.nn.L1Loss,
        "MarginRankingLoss": torch.nn.MarginRankingLoss,
        "MSELoss": torch.nn.MSELoss,
        "MultiLabelMarginLoss": torch.nn.MultiLabelMarginLoss,
        "MultiLabelSoftMarginLoss": torch.nn.MultiLabelSoftMarginLoss,
        "MultiMarginLoss": torch.nn.MultiMarginLoss,
        "NLLLoss": torch.nn.NLLLoss,
        "NLLLoss2d": torch.nn.NLLLoss2d,
        "PoissonNLLLoss": torch.nn.PoissonNLLLoss,
        "SmoothL1Loss": torch.nn.SmoothL1Loss,
        "SoftMarginLoss": torch.nn.SoftMarginLoss,
        "TripletMarginLoss": torch.nn.TripletMarginLoss,
    },
)

registry.register_all_with_type(
    ActivationFn,
    {
        "ELU": torch.nn.ELU,
        "Hardshrink": torch.nn.Hardshrink,
        "Hardsigmoid": torch.nn.Hardsigmoid,
        "Hardtanh": torch.nn.Hardtanh,
        "Hardswish": torch.nn.Hardswish,
        "LeakyReLU": torch.nn.LeakyReLU,
        "LogSigmoid": torch.nn.LogSigmoid,
        "MultiheadAttention": torch.nn.MultiheadAttention,
        "PReLU": torch.nn.PReLU,
        "ReLU": torch.nn.ReLU,
        "ReLU6": torch.nn.ReLU6,
        "RReLU": torch.nn.RReLU,
        "SELU": torch.nn.SELU,
        "CELU": torch.nn.CELU,
        "GELU": torch.nn.GELU,
        "Sigmoid": torch.nn.Sigmoid,
        "SiLU": torch.nn.SiLU,
        "Softplus": torch.nn.Softplus,
        "Softshrink": torch.nn.Softshrink,
        "Softsign": torch.nn.Softsign,
        "Tanh": torch.nn.Tanh,
        "Tanhshrink": torch.nn.Tanhshrink,
        "Threshold": torch.nn.Threshold,
        "GLU": torch.nn.GLU,
        "Softmin": torch.nn.Softmin,
        "Softmax": torch.nn.Softmax,
        "Softmax2d": torch.nn.Softmax2d,
        "LogSoftmax": torch.nn.LogSoftmax,
        "AdaptiveLogSoftmaxWithLoss": torch.nn.AdaptiveLogSoftmaxWithLoss,
    },
)

registry.register_classes(
    {
        f"torch.nn.{x}": getattr(torch.nn, x)
        for x in dir(torch.nn)
        if inspect.isclass(getattr(torch.nn, x))
        and issubclass(getattr(torch.nn, x), torch.nn.Module)
    }
)
