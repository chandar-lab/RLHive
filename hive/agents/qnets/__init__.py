import torch
from hive.agents.qnets.mlp import SimpleMLP
from hive.agents.qnets.rainbow_mlp import ComplexMLP, DistributionalMLP
from hive.agents.qnets.conv import SimpleConvModel
from hive.agents.qnets.atari import NatureAtariDQNModel
from hive.agents.qnets.minatar import MinAtarDQNModel
from hive.utils.utils import create_class_constructor

get_qnet = create_class_constructor(
    torch.nn.Module,
    {
        "SimpleMLP": SimpleMLP,
        "ComplexMLP": ComplexMLP,
        "DistributionalMLP": DistributionalMLP,
        "SimpleConvModel": SimpleConvModel,
        "NatureAtariDQNModel": NatureAtariDQNModel,
    },
)
