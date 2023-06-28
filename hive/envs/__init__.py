from hive.envs.base import BaseEnv, ParallelEnv
from hive.envs.env_spec import EnvSpec
from hive.envs.gym.gym_env import GymEnv

try:
    from hive.envs.pettingzoo import PettingZooEnv
except ImportError:
    PettingZooEnv = None


from hive.utils.registry import registry

registry.register_all(
    BaseEnv,
    {
        "GymEnv": GymEnv,
        "PettingZooEnv": PettingZooEnv,
    },
)

get_env = getattr(registry, f"get_{BaseEnv.type_name()}")
