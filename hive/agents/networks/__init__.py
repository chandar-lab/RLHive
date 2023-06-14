from hive.agents.networks.conv import ConvNetwork
from hive.agents.networks.mlp import MLPNetwork
from hive.utils.registry import registry

registry.register_classes(
    {
        "ConvNetwork": ConvNetwork,
        "MLPNetwork": MLPNetwork,
    },
)
