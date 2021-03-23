from hive.envs.env_spec import EnvSpec
from hive.envs.base import BaseEnv
from hive.envs.gym_env import GymEnv
from hive.envs.minigrid import MiniGridSingleAgent

from hive.utils.utils import create_class_constructor

get_env = create_class_constructor(BaseEnv, {"GymEnv": GymEnv, "MiniGridSingleAgent": MiniGridSingleAgent})
