import gym
import numpy as np
from hive.envs import GymEnv, ParallelEnv
from hive.envs.wrappers.gym_wrappers import FlattenWrapper, PermuteImageWrapper
from marlgrid import envs
from gym.wrappers.compatibility import EnvCompatibility

from numpy.random._generator import Generator


class MyGenerator(Generator):
    def randint(
        self: Generator,
        low: int,
        high: int,
        size=None,
        dtype="l",
        endpoint: bool = False,
    ):
        """Replacement for `numpy.random.Generator.randint` that uses the
        `Generator.integers` method instead of `Generator.random_integers`
        which is deprecated."""
        return self.integers(low, high, size=size, dtype=dtype, endpoint=endpoint)


def _patched_np_random(seed: int = None):
    """Replacement for `gym.utils.seeding.np_random` that uses the
    `MyGenerator` class instead of `numpy.random.Generator`.
    MyGenerator has a `.randint` method so the old code from marlgrid
    can still work."""
    from gym import error

    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        raise error.Error(f"Seed must be a non-negative integer or omitted, not {seed}")
    seed_seq = np.random.SeedSequence(seed)
    np_seed = seed_seq.entropy
    rng = MyGenerator(np.random.PCG64(seed_seq))
    return rng, np_seed


gym.utils.seeding.np_random = _patched_np_random

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
