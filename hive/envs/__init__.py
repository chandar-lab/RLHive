from hive.envs.base import BaseEnv
from hive.envs.gym.gym_env import GymEnv

try:
    from hive.envs.marlgrid import MarlGridEnv
except ImportError:
    MarlGridEnv = None

try:
    from hive.envs.pettingzoo import PettingZooEnv
except ImportError:
    PettingZooEnv = None

from hive.utils.registry import registry

registry.register_all(
    BaseEnv,
    {
        "GymEnv": GymEnv,
        "MarlGridEnv": MarlGridEnv,
        "PettingZooEnv": PettingZooEnv,
    },
)

get_env = getattr(registry, f"get_{BaseEnv.type_name()}")
