import torch
from hive.agents.qnets.mlp import SimpleMLP
from hive.agents.qnets.nature_dqn import NatureDQNModel
from hive.utils.utils import create_class_constructor

get_qnet = create_class_constructor(
    torch.nn.Module,
    {
        "SimpleMLP": SimpleMLP,
        "NatureDQNModel": NatureDQNModel,
    },
)
