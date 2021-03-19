import gym_minigrid
from gym_minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, \
    RGBImgPartialObsWrapper, ImgObsWrapper

from hive.envs.gym_env import GymEnv
from hive.envs.env_spec import EnvSpec


class GymMiniGridImageEnv(GymEnv):
    """
    Class for loading gym-minigrid environments in which the observation is an image.
    """

    def __init__(
            self,
            env_name,
            fully_observable=True,
            only_image=True
    ):
        """
        Args:
            env_name: Name of the environment
            fully_observable: True if fully observable
            only_image: True if mission should be removed
        """

        super(GymMiniGridImageEnv, self).__init__(env_name=env_name)

        if fully_observable:
            self._env = FullyObsWrapper(self._env)
            self._env = RGBImgObsWrapper(self._env)

        else:
            self._env = RGBImgPartialObsWrapper(self._env)

        if only_image:
            self._env = ImgObsWrapper(self._env)

        self.env_spec = EnvSpec(env_name=env_name,
                                obs_dim=self._env.observation_space.shape,
                                act_dim=self._env.action_space.n)

    def render(self, mode='rgb_array'):
        # TODO
        pass
