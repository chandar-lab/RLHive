from marlgrid import envs
import gym
from hive.envs import ParallelEnv, GymEnv
from hive.envs.wrappers import FlattenWrapper


class MarlGridEnv(ParallelEnv, GymEnv):
    def create_env(self, env_name, randomize_seed=True, flatten=True, **kwargs):
        if env_name is None:
            self._env = envs.env_from_config(kwargs, randomize_seed=randomize_seed)
        else:
            super().create_env(env_name, **kwargs)

        if flatten:
            self._env = FlattenWrapper(self._env)

    def create_env_spec(self, name, **kwargs):
        return super().create_env_spec(
            name if name is not None else f"Marlgrid_{str(kwargs)}", **kwargs
        )
