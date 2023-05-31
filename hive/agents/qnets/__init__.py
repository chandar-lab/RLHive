from hive.utils.registry import registry
from hive.agents.qnets.atari import NatureAtariDQNModel
from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.conv import ConvNetwork
from hive.agents.qnets.mlp import MLPNetwork
from hive.agents.qnets.sequence_models import (
    SequenceModel,
    SequenceFn,
)

registry.register_all(
    FunctionApproximator,
    {
        "ConvNetwork": ConvNetwork,
        "MLPNetwork": MLPNetwork,
        "NatureAtariDQNModel": NatureAtariDQNModel,
    },
)

get_qnet = getattr(registry, f"get_{FunctionApproximator.type_name()}")
