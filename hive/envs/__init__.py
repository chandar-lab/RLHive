from hive.envs.base import BaseEnv, ParallelEnv
from hive.envs.env_spec import EnvSpec
from hive.envs.gym.gym_env import GymEnv
from hive.utils.registry import registry

registry.register_class("GymEnv", GymEnv)

try:
    from hive.envs.pettingzoo import PettingZooEnv

    registry.register_class("PettingZooEnv", PettingZooEnv)
except ImportError:
    pass
