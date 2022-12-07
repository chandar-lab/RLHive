from typing import List

import gym
import numpy as np

from hive.envs.base import BaseEnv
from hive.envs.env_spec import EnvSpec
from hive.utils.registry import Registrable, registry


class GymEnv(BaseEnv):
    """
    Class for loading gym environments.
    """

    def __init__(self, env_name, num_players=1, **kwargs):
        """
        Args:
            env_name (str): Name of the environment (NOTE: make sure it is available
                in gym.envs.registry.all())
            num_players (int): Number of players for the environment.
            kwargs: Any arguments you want to pass to :py:meth:`create_env` or
                :py:meth:`create_env_spec` can be passed as keyword arguments to this
                constructor.
        """
        self.create_env(env_name, **kwargs)
        super().__init__(self.create_env_spec(env_name, **kwargs), num_players)

    def create_env(self, env_name, **kwargs):
        """Function used to create the environment. Subclasses can override this method
        if they are using a gym style environment that needs special logic.

        Args:
            env_name (str): Name of the environment
        """
        env = gym.make(env_name)

        wrapper_config = kwargs.get("wrappers", None)
        if isinstance(wrapper_config, list) and len(wrapper_config) > 0:
            wrapper_config = {
                "name": "CompositeEnvWrapper",
                "kwargs": {"wrapper_list": wrapper_config},
            }
        if wrapper_config is not None:
            wrapper_fn, _ = get_wrapper(wrapper_config, "wrappers")
            env = wrapper_fn(env)

        self._env = env

    def create_env_spec(self, env_name, **kwargs):
        """Function used to create the specification. Subclasses can override this method
        if they are using a gym style environment that needs special logic.

        Args:
            env_name (str): Name of the environment
        """
        if isinstance(self._env.observation_space, gym.spaces.Tuple):
            observation_spaces = self._env.observation_space.spaces
        else:
            observation_spaces = [self._env.observation_space]
        if isinstance(self._env.action_space, gym.spaces.Tuple):
            action_spaces = self._env.action_space.spaces
        else:
            action_spaces = [self._env.action_space]

        return EnvSpec(
            env_name=env_name,
            observation_space=observation_spaces,
            action_space=action_spaces,
        )

    def reset(self):
        observation = self._env.reset()
        return observation, self._turn

    def step(self, action):
        observation, reward, done, info = self._env.step(action)
        self._turn = (self._turn + 1) % self._num_players
        return observation, reward, done, self._turn, info

    def render(self, mode="rgb_array"):
        return self._env.render(mode=mode)

    def seed(self, seed=None):
        self._env.seed(seed=seed)

    def close(self):
        self._env.close()


class EnvWrapper(Registrable):
    """A wrapper for callables that produce environment wrappers.

    These wrapped callables can be partially initialized through configuration
    files or command line arguments.
    """

    @classmethod
    def type_name(cls):
        """
        Returns:
            "env_wrapper"
        """
        return "env_wrapper"


class CompositeEnvWrapper(gym.Wrapper):
    """This Logger aggregates multiple env wrappers."""

    def __init__(self, env, wrapper_list: List[EnvWrapper]):
        for wrapper in wrapper_list:
            env = wrapper(env)
        super().__init__(env)


registry.register_all(
    EnvWrapper,
    {
        "RecordEpisodeStatistics": gym.wrappers.RecordEpisodeStatistics,
        "ClipAction": gym.wrappers.ClipAction,
        "NormalizeObservation": gym.wrappers.NormalizeObservation,
        "TransformObservation": gym.wrappers.TransformObservation,
        "NormalizeReward": gym.wrappers.NormalizeReward,
        "TransformReward": gym.wrappers.TransformReward,
        "CompositeEnvWrapper": CompositeEnvWrapper,
    },
)

get_wrapper = getattr(registry, f"get_{EnvWrapper.type_name()}")
