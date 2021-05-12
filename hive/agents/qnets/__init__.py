import torch
from hive.agents.qnets.mlp import SimpleMLP
from hive.agents.qnets.rainbow_mlp import ComplexMLP, DistributionalMLP
from hive.utils.utils import create_class_constructor

get_qnet = create_class_constructor(torch.nn.Module, {"SimpleMLP": SimpleMLP,
                                                      "ComplexMLP": ComplexMLP,
                                                      "DistributionalMLP": DistributionalMLP
                                                      })
