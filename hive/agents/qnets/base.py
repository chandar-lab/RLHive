import torch
from typing import NewType

FunctionApproximator = NewType("FunctionApproximator", torch.nn.Module)
