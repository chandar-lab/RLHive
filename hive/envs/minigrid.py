import gym_minigrid

from gym_minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, FlatObsWrapper, \
    RGBImgPartialObsWrapper, ImgObsWrapper

from hive.envs.gym_env import GymEnv
from hive.envs.env_spec import EnvSpec
from hive.envs.wrappers.gym_wrappers import FlattenWrapper


class MiniGridSingleAgent(GymEnv):
    """
    Class for loading MiniGrid environments (https://github.com/maximecb/gym-minigrid).
    """

    def __init__(
            self,
            env_name,
            rgb_obs=False,
            flattened_obs=True,
            fully_observable=True,
            use_mission=False
    ):
        """
        Args:
            env_name: Name of the environment
            rgb_obs: True if observations should be rgb-like images
            flattened_obs: True for flattening the observation into one dimensional vector
            fully_observable: True if fully observable
            use_mission: True if mission should be in the observation, in which case:
             if using rgb_obs or non-rgb non-flattened grid, the observation is a dict of keys, image and mission.
             if using non-rgb flattened observations, then the observation has the mission encoded in it.
        """

        super(MiniGridSingleAgent, self).__init__(env_name=env_name)

        if fully_observable:
            self._env = FullyObsWrapper(self._env)
            if rgb_obs:
                self._env = RGBImgObsWrapper(self._env)
        elif rgb_obs:
            self._env = RGBImgPartialObsWrapper(self._env)

        if not use_mission:
            self._env = ImgObsWrapper(self._env)
            if flattened_obs:
                self._env = FlattenWrapper(self._env)
        elif not rgb_obs:
            # Encode the mission into observation vector
            self._env = FlatObsWrapper(self._env)

        self.env_spec = EnvSpec(env_name=env_name,
                                obs_dim=self._env.observation_space.shape,
                                act_dim=self._env.action_space.n)

    def render(self, mode='rgb_array'):
        # TODO
        pass
