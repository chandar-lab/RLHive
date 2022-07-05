import gym_minigrid
from gym_minigrid.wrappers import (
    FlatObsWrapper,
    FullyObsWrapper,
    ImgObsWrapper,
    RGBImgObsWrapper,
    RGBImgPartialObsWrapper,
)

from hive.envs.gym_env import GymEnv
from hive.envs.wrappers.gym_wrappers import FlattenWrapper, PermuteImageWrapper


class MiniGridEnv(GymEnv):
    """
    Class for loading MiniGrid environments (https://github.com/maximecb/gym-minigrid).
    """

    def create_env(
        self,
        env_name,
        rgb_obs=True,
        flattened_obs=False,
        fully_observable=True,
        use_mission=False,
    ):
        """
        Args:
            env_name (str): Name of the environment.
            rgb_obs (bool): True if observations should be rgb-like images
            flattened_obs (bool): True for flattening the observation into one
                dimensional vector.
            fully_observable (bool): Whether to make the environment fully observable.
            use_mission (bool): Whether mission should be in the observation, in which
                case if using non-flattened grid, the observation is a dict of keys,
                image and mission. If using flattened observations, then the
                observation has the mission encoded in it.
        """

        super().create_env(env_name)

        if fully_observable:
            self._env = FullyObsWrapper(self._env)
            if rgb_obs:
                self._env = RGBImgObsWrapper(self._env)
        elif rgb_obs:
            self._env = RGBImgPartialObsWrapper(self._env)

        if not use_mission:
            self._env = PermuteImageWrapper(ImgObsWrapper(self._env))
            if flattened_obs:
                self._env = FlattenWrapper(self._env)
        elif flattened_obs:
            # Encode the mission into observation vector
            self._env = FlatObsWrapper(self._env)

    def render(self, mode="rgb_array"):
        # TODO
        pass
