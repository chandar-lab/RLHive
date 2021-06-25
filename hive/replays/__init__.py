from hive.replays.replay_buffer import CircularReplayBuffer, BaseReplayBuffer
from hive.replays.efficient_replay import EfficientCircularBuffer
from hive import registry


registry.register_all(
    BaseReplayBuffer,
    {
        "CircularReplayBuffer": CircularReplayBuffer,
        "EfficientCircularBuffer": EfficientCircularBuffer,
    },
)

get_replay = getattr(registry, f"get_{BaseReplayBuffer.type_name()}")
