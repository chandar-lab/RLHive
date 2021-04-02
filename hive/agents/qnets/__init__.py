import torch
from hive.agents.qnets.mlp import SimpleMLP, DuelingMLP
from hive.utils.utils import create_class_constructor

get_qnet = create_class_constructor(torch.nn.Module, {"SimpleMLP": SimpleMLP,
                                                      "DuelingMLP": DuelingMLP
                                                      })
