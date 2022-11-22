import gym
from hive.envs import GymEnv, ParallelEnv
from hive.envs.wrappers.gym_wrappers import FlattenWrapper, PermuteImageWrapper
from marlgrid import envs
from gym.wrappers.compatibility import EnvCompatibility

from hive.utils.utils import _patched_np_random

gym.utils.seeding.np_random = _patched_np_random


class MarlGridEnv(ParallelEnv, GymEnv):
    """MarlGrid environment from https://github.com/kandouss/marlgrid/.

    The environment can either be initialized with the name of a preregistered
    environment from
    https://github.com/kandouss/marlgrid/blob/master/marlgrid/envs/__init__.py,
    or can be created using a config. See the original repo for details.
    """

    def create_env(self, env_name, randomize_seed=True, flatten=False, **kwargs):
        """
        Args:
            env_name: The name of the environment.
            randomize_seed: Whether to use a random random seed for the environment.
            flatten: Whether to flatten the observations.
        """
        if env_name is None:
            self._env = envs.env_from_config(kwargs, randomize_seed=randomize_seed)
            self._env = EnvCompatibility(self._env)
        else:
            super().create_env(env_name, apply_api_compatibility=True, **kwargs)

        self._env = PermuteImageWrapper(self._env)
        if flatten:
            self._env = FlattenWrapper(self._env)

    def create_env_spec(self, name, **kwargs):
        return super().create_env_spec(
            name if name is not None else f"Marlgrid_{str(kwargs)}", **kwargs
        )

    def reset(self):
        obs = super().reset()
        return obs
