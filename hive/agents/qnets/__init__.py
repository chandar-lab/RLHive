from hive import registry
from hive.agents.qnets.atari import NatureAtariDQNModel
from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.conv import SimpleConvModel
from hive.agents.qnets.mlp import SimpleMLP, DiscObsSimpleMLP
from hive.agents.qnets.rainbow_mlp import ComplexMLP, DistributionalMLP

registry.register_all(
    FunctionApproximator,
    {
        "SimpleMLP": FunctionApproximator(SimpleMLP),
        "DiscObsSimpleMLP": FunctionApproximator(DiscObsSimpleMLP),
        "ComplexMLP": FunctionApproximator(ComplexMLP),
        "DistributionalMLP": FunctionApproximator(DistributionalMLP),
        "SimpleConvModel": FunctionApproximator(SimpleConvModel),
        "NatureAtariDQNModel": FunctionApproximator(NatureAtariDQNModel),
    },
)

get_qnet = getattr(registry, f"get_{FunctionApproximator.type_name()}")
