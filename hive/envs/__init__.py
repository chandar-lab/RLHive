import os

from hive.envs.env_spec import EnvSpec
from hive.envs.base import BaseEnv, ParallelEnv
from hive.envs.gym_env import GymEnv
from hive.envs.minigrid import MiniGridEnv

try:
    from hive.envs.minigrid import MiniGridEnv
except ImportError:
    MiniGridEnv = None

try:
    from hive.envs.marlgrid import MarlGridEnv
except ImportError:
    MarlGridEnv = None

try:
    from hive.envs.multiagent_discrete import MultiAgentDiscEnv
except ImportError:
    MarlGridEnv = None

from hive.utils.utils import create_class_constructor

get_env = create_class_constructor(
    BaseEnv,
    {
        "GymEnv": GymEnv,
        "MiniGridEnv": MiniGridEnv,
        "MarlGridEnv": MarlGridEnv,
        "MultiAgentDiscEnv": MultiAgentDiscEnv,
    },
)
