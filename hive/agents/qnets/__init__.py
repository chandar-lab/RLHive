import torch
from hive.agents.qnets.mlp import SimpleMLP, DistributionalMLP
from hive.utils.utils import create_class_constructor

get_qnet = create_class_constructor(torch.nn.Module, {"SimpleMLP": SimpleMLP,
                                                      "DistributionalMLP": DistributionalMLP
                                                      })
