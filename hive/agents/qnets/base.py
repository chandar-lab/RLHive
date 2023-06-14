from typing import NewType

import torch

torch.nn.Module = NewType("torch.nn.Module", torch.nn.Module)
