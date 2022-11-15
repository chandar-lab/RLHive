import operator
from functools import reduce

import gym
import numpy as np


class FlattenWrapper(gym.core.ObservationWrapper):
    """
    Flatten the observation to one dimensional vector.
    """

    def __init__(self, env):
        super().__init__(env)

        if isinstance(env.observation_space, gym.spaces.Tuple):
            self.observation_space = gym.spaces.Tuple(
                tuple(
                    gym.spaces.Box(
                        low=space.low.flatten(),
                        high=space.high.flatten(),
                        shape=(reduce(operator.mul, space.shape, 1),),
                        dtype=space.dtype,
                    )
                    for space in env.observation_space
                )
            )
            self._is_tuple = True
        else:
            self.observation_space = gym.spaces.Box(
                low=env.observation_space.low.flatten(),
                high=env.observation_space.high.flatten(),
                shape=(reduce(operator.mul, env.observation_space.shape, 1),),
                dtype=env.observation_space.dtype,
            )
            self._is_tuple = False

    def observation(self, obs):
        if self._is_tuple:
            return tuple(o.flatten() for o in obs)
        else:
            return obs.flatten()


class PermuteImageWrapper(gym.core.ObservationWrapper):
    """Changes the image format from HWC to CHW"""

    def __init__(self, env):
        super().__init__(env)

        if isinstance(env.observation_space, gym.spaces.Tuple):
            self.observation_space = gym.spaces.Tuple(
                tuple(
                    gym.spaces.Box(
                        low=np.transpose(space.low, [2, 1, 0]),
                        high=np.transpose(space.high, [2, 1, 0]),
                        shape=(space.shape[-1],) + space.shape[:-1],
                        dtype=space.dtype,
                    )
                    for space in env.observation_space
                )
            )
            self._is_tuple = True
        else:
            self.observation_space = gym.spaces.Box(
                low=np.transpose(env.observation_space.low, [2, 1, 0]),
                high=np.transpose(env.observation_space.high, [2, 1, 0]),
                shape=(env.observation_space.shape[-1],)
                + env.observation_space.shape[:-1],
                dtype=env.observation_space.dtype,
            )
            self._is_tuple = False

    def observation(self, obs):
        if self._is_tuple:
            return tuple(np.transpose(o, [2, 1, 0]) for o in obs)
        else:
            return np.transpose(obs, [2, 1, 0])


class FlickeringWrapper(gym.core.ObservationWrapper):
    """Fully obscure the image with certain probablity."""

    def __init__(self, env, flicker_prob=0.5):
        super().__init__(env)

        self.flicker_prob = flicker_prob
        if isinstance(env.observation_space, gym.spaces.Tuple):
            self._is_tuple = True
            self.obscured_obs = np.zeros(
                shape=env.observation_space[0].shape,
                dtype=np.uint8,
            )
        else:
            self._is_tuple = False
            self.obscured_obs = np.zeros(
                shape=env.observation_space.shape,
                dtype=np.uint8,
            )

    def observation(self, obs):
        if not np.random.binomial(n=1, p=self.flicker_prob):
            return obs

        if self._is_tuple:
            return tuple(self.obscured_obs for _ in obs)
        else:
            return self.obscured_obs
