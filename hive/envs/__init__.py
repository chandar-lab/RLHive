from hive.utils.registry import registry

from hive.envs.base import BaseEnv, ParallelEnv
from hive.envs.env_spec import EnvSpec
from hive.envs.gym.gym_env import GymEnv

registry.register("GymEnv", GymEnv, BaseEnv)

try:
    from hive.envs.marlgrid import MarlGridEnv

    registry.register("MarlGridEnv", MarlGridEnv, BaseEnv)
except ImportError:
    pass
try:
    from hive.envs.pettingzoo import PettingZooEnv

    registry.register("PettingZooEnv", PettingZooEnv, BaseEnv)
except ImportError:
    pass
