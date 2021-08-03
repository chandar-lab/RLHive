import torch
from hive.agents.qnets.mlp import SimpleMLP
from hive.agents.qnets.rainbow_mlp import ComplexMLP, DistributionalMLP
from hive.agents.qnets.conv import SimpleConvModel
from hive.agents.qnets.atari import NatureAtariDQNModel
from hive.agents.qnets.hanabi_rainbow_mlp import (
    ComplexHanabiMLP,
    DistributionalHanabiMLP,
)
from hive.utils.utils import create_class_constructor

registry.register_all(
    FunctionApproximator,
    {
        "SimpleMLP": FunctionApproximator(SimpleMLP),
        "ComplexMLP": FunctionApproximator(ComplexMLP),
        "DistributionalMLP": FunctionApproximator(DistributionalMLP),
        "SimpleConvModel": FunctionApproximator(SimpleConvModel),
        "NatureAtariDQNModel": FunctionApproximator(NatureAtariDQNModel),
        "ComplexHanabiMLP": FunctionApproximator(ComplexHanabiMLP),
        "DistributionalHanabiMLP": FunctionApproximator(DistributionalHanabiMLP),

    },
)

get_qnet = getattr(registry, f"get_{FunctionApproximator.type_name()}")
