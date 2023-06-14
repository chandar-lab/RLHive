from hive.replays.circular_replay import CircularReplayBuffer, SimpleReplayBuffer
from hive.replays.on_policy_replay import OnPolicyReplayBuffer
from hive.replays.prioritized_replay import PrioritizedReplayBuffer
from hive.replays.recurrent_replay import RecurrentReplayBuffer
from hive.replays.replay_buffer import Alignment, BaseReplayBuffer, ReplayItemSpec
from hive.utils.registry import registry

registry.register_classes(
    {
        "CircularReplayBuffer": CircularReplayBuffer,
        "OnPolicyReplayBuffer": OnPolicyReplayBuffer,
        "PrioritizedReplayBuffer": PrioritizedReplayBuffer,
        "RecurrentReplayBuffer": RecurrentReplayBuffer,
        "SimpleReplayBuffer": SimpleReplayBuffer,
    },
)
