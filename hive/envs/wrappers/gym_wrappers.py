from functools import reduce
import operator
import gym


class FlattenWrapper(gym.core.ObservationWrapper):
    """
    Flatten the observation to one dimensional vector.
    """

    def __init__(self, env):
        super().__init__(env)

        img_size = reduce(operator.mul, env.observation_space.shape, 1)

        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.flatten(),
            high=env.observation_space.high.flatten(),
            shape=(img_size,),
            dtype=env.observation_space.dtype
        )

    def observation(self, obs):
        return obs.flatten()
