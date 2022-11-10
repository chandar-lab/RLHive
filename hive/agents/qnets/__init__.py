import logging

from hive.utils.registry import registry
from hive.agents.qnets.atari import NatureAtariDQNModel
from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.conv import ConvNetwork
from hive.agents.qnets.mlp import MLPNetwork
from hive.agents.qnets.rnn import ConvRNNNetwork

registry.register_all(
    FunctionApproximator,
    {
        "ConvNetwork": ConvNetwork,
        "ConvRNNNetwork": ConvRNNNetwork,
        "MLPNetwork": MLPNetwork,
        "NatureAtariDQNModel": NatureAtariDQNModel,
    },
)

logging.info("Registered function approximators.")
get_qnet = getattr(registry, f"get_{FunctionApproximator.type_name()}")
