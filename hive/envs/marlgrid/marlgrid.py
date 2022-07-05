from hive.envs import GymEnv, ParallelEnv
from hive.envs.wrappers.gym_wrappers import FlattenWrapper, PermuteImageWrapper
from marlgrid import envs


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
        else:
            super().create_env(env_name, **kwargs)

        self._env = PermuteImageWrapper(self._env)
        if flatten:
            self._env = FlattenWrapper(self._env)

    def create_env_spec(self, name, **kwargs):
        return super().create_env_spec(
            name if name is not None else f"Marlgrid_{str(kwargs)}", **kwargs
        )
