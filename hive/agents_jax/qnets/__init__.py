from hive.utils.registry import registry
from hive.agents_jax.qnets.conv import JaxConvNetwork
from hive.agents_jax.qnets.mlp import JaxMLPNetwork
from hive.agents_jax.qnets.networks import JaxNatureAtariDQNModel
from hive.agents.qnets.base import FunctionApproximator

registry.register_all(
    FunctionApproximator,
    {
        "JaxMLPNetwork": FunctionApproximator(JaxMLPNetwork),
        "JaxConvNetwork": FunctionApproximator(JaxConvNetwork),
        "JaxNatureAtariDQNModel": FunctionApproximator(JaxNatureAtariDQNModel),
    },
)

get_qnet = getattr(registry, f"get_{FunctionApproximator.type_name()}")
