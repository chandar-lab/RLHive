import torch

from hive.agents.qnets.conv import ConvNetwork
from hive.agents.qnets.mlp import MLPNetwork
from hive.agents.qnets.sequence_models import SequenceFn, SequenceModel
from hive.utils.registry import registry

registry.register_all(
    torch.nn.Module,
    {
        "ConvNetwork": ConvNetwork,
        "MLPNetwork": MLPNetwork,
    },
)
