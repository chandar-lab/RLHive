from hive.envs.env_spec import EnvSpec
from hive.envs.base import BaseEnv, ParallelEnv
from hive.envs.gym_env import GymEnv
from hive.envs.probe.probe import Probe1, Probe2, Probe3, Probe4, Probe5

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

from hive.utils.utils import create_class_constructor

get_env = create_class_constructor(
    BaseEnv,
    {
        "GymEnv": GymEnv,
        "MiniGridEnv": MiniGridEnv,
        "MarlGridEnv": MarlGridEnv,
        "AtariEnv": AtariEnv,
        "Probe1": Probe1,
        "Probe2": Probe2,
        "Probe3": Probe3,
        "Probe4": Probe4,
        "Probe5": Probe5,
    },
)
