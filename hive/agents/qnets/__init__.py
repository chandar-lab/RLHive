from hive import registry
from hive.agents.qnets.atari import NatureAtariDQNModel
from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.conv import SimpleConvModel
from hive.agents.qnets.mlp import SimpleMLP
from hive.agents.qnets.rainbow_mlp import ComplexMLP, DistributionalMLP

registry.register_all(
    FunctionApproximator,
    {
        "SimpleMLP": SimpleMLP,
        "ComplexMLP": ComplexMLP,
        "DistributionalMLP": DistributionalMLP,
        "SimpleConvModel": SimpleConvModel,
        "NatureAtariDQNModel": NatureAtariDQNModel,
    },
)

get_qnet = getattr(registry, f"get_{FunctionApproximator.type_name()}")
