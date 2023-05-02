from hive.envs.base import BaseEnv, ParallelEnv
from hive.envs.env_spec import EnvSpec
from hive.envs.gym_env import GymEnv
from minigrid import (
    FlatObsWrapper,
    FullyObsWrapper,
    ImgObsWrapper,
    RGBImgObsWrapper,
    RGBImgPartialObsWrapper,
)

try:
    from hive.envs.atari import AtariEnv
except ImportError:
    AtariEnv = None

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
        "minigrid.FullyObsWrapper": FullyObsWrapper,
        "minigrid.FlatObsWrapper": FlatObsWrapper,
        "minigrid.ImgObsWrapper": ImgObsWrapper,
        "minigrid.RGBImgObsWrapper": RGBImgObsWrapper,
        "minigrid.RGBImgPartialObsWrapper": RGBImgPartialObsWrapper,
        "MarlGridEnv": MarlGridEnv,
        "AtariEnv": AtariEnv,
        "PettingZooEnv": PettingZooEnv,
    },
)

get_env = getattr(registry, f"get_{BaseEnv.type_name()}")
