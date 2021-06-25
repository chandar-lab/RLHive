from hive.envs.base import BaseEnv, ParallelEnv
from hive.envs.env_spec import EnvSpec
from hive.envs.gym_env import GymEnv

try:
    from hive.envs.minigrid import MiniGridEnv
except ImportError:
    MiniGridEnv = None

try:
    from hive.envs.atari import AtariEnv
except ImportError:
    AtariEnv = None

try:
    from hive.envs.marlgrid import MarlGridEnv
except ImportError:
    MarlGridEnv = None

from hive import registry

registry.register_all(
    BaseEnv,
    {
        "GymEnv": GymEnv,
        "MiniGridEnv": MiniGridEnv,
        "MarlGridEnv": MarlGridEnv,
        "AtariEnv": AtariEnv,
    },
)

get_env = getattr(registry, f"get_{BaseEnv.type_name()}")
