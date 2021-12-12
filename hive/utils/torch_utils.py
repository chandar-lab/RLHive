import numpy as np
import torch
from torch import optim

from hive.utils.registry import registry
from hive.utils.utils import LossFn, OptimizerFn


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
        "RMSpropTF": OptimizerFn(RMSpropTF),
        "Rprop": OptimizerFn(optim.Rprop),
        "SGD": OptimizerFn(optim.SGD),
        "SparseAdam": OptimizerFn(optim.SparseAdam),
    },
)

registry.register_all(
    LossFn,
    {
        "BCELoss": LossFn(torch.nn.BCELoss),
        "BCEWithLogitsLoss": LossFn(torch.nn.BCEWithLogitsLoss),
        "CosineEmbeddingLoss": LossFn(torch.nn.CosineEmbeddingLoss),
        "CrossEntropyLoss": LossFn(torch.nn.CrossEntropyLoss),
        "CTCLoss": LossFn(torch.nn.CTCLoss),
        "HingeEmbeddingLoss": LossFn(torch.nn.HingeEmbeddingLoss),
        "KLDivLoss": LossFn(torch.nn.KLDivLoss),
        "L1Loss": LossFn(torch.nn.L1Loss),
        "MarginRankingLoss": LossFn(torch.nn.MarginRankingLoss),
        "MSELoss": LossFn(torch.nn.MSELoss),
        "MultiLabelMarginLoss": LossFn(torch.nn.MultiLabelMarginLoss),
        "MultiLabelSoftMarginLoss": LossFn(torch.nn.MultiLabelSoftMarginLoss),
        "MultiMarginLoss": LossFn(torch.nn.MultiMarginLoss),
        "NLLLoss": LossFn(torch.nn.NLLLoss),
        "NLLLoss2d": LossFn(torch.nn.NLLLoss2d),
        "PoissonNLLLoss": LossFn(torch.nn.PoissonNLLLoss),
        "SmoothL1Loss": LossFn(torch.nn.SmoothL1Loss),
        "SoftMarginLoss": LossFn(torch.nn.SoftMarginLoss),
        "TripletMarginLoss": LossFn(torch.nn.TripletMarginLoss),
    },
)

get_optimizer_fn = getattr(registry, f"get_{OptimizerFn.type_name()}")
get_loss_fn = getattr(registry, f"get_{LossFn.type_name()}")
