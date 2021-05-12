from hive.replays.replay_buffer import CircularReplayBuffer, BaseReplayBuffer
from hive.replays.efficient_replay import EfficientCircularBuffer
from hive.utils.utils import create_class_constructor


get_replay = create_class_constructor(
    BaseReplayBuffer,
    {
        "CircularReplayBuffer": CircularReplayBuffer,
        "EfficientCircularBuffer": EfficientCircularBuffer,
    },
)

