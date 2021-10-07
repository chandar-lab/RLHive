from hive import registry
from hive.replays.efficient_replay import EfficientCircularBuffer
from hive.replays.prioritized_replay import PrioritizedReplayBuffer
from hive.replays.replay_buffer import BaseReplayBuffer, CircularReplayBuffer
from hive.replays.hanabi_buffer import HanabiBuffer

registry.register_all(
    BaseReplayBuffer,
    {
        "CircularReplayBuffer": CircularReplayBuffer,
        "EfficientCircularBuffer": EfficientCircularBuffer,
        "PrioritizedReplayBuffer": PrioritizedReplayBuffer,
        "HanabiBuffer": HanabiBuffer,
    },
)

get_replay = getattr(registry, f"get_{BaseReplayBuffer.type_name()}")
