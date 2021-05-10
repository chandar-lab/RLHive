import os
from hive.envs.env_spec import EnvSpec
from hive.envs.base import BaseEnv, ParallelEnv
from hive.envs.gym_env import GymEnv
from hive.envs.minigrid import MiniGridEnv
import hive.envs.ma_envs

if "GITHUB_CI" in os.environ:
    MarlGridEnv = None
    MultiAgentDiscEnv = None
else:
    from hive.envs.marlgrid import MarlGridEnv
    from hive.envs.multiagent_discrete import MultiAgentDiscEnv

from hive.utils.utils import create_class_constructor

get_env = create_class_constructor(
    BaseEnv, {"GymEnv": GymEnv, "MiniGridEnv": MiniGridEnv, "MarlGridEnv": MarlGridEnv, "MultiAgentDiscEnv": MultiAgentDiscEnv}
)
