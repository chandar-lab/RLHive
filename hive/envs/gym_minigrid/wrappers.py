from functools import reduce
import operator
import gym


class FlatObsNoMissionWrapper(gym.core.ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array
    """

    def __init__(self, env):
        super().__init__(env)

        img_space = env.observation_space.spaces['image']
        img_size = reduce(operator.mul, img_space.shape, 1)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(img_size,),
            dtype='uint8'
        )

    def observation(self, obs):
        image = obs['image']
        obs = image.flatten()
        return obs