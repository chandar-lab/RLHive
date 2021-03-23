import gym_minigrid

from gym_minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, FlatObsWrapper, \
    RGBImgPartialObsWrapper, ImgObsWrapper

from hive.envs.gym_env import GymEnv
from hive.envs.env_spec import EnvSpec
from hive.envs.wrappers.minigrid_wrappers import FlatObsNoMissionWrapper


class MiniGridSingleAgent(GymEnv):
    """
    Class for loading MiniGrid environments (https://github.com/maximecb/gym-minigrid).
    """

    def __init__(
            self,
            env_name,
            image_based_obs=False,
            fully_observable=True,
            use_mission=False
    ):
        """
        Args:
            env_name: Name of the environment
            image_based_obs: True if observations should be images, otherwise
            they are one dimensional vectors (flatten observation)
            fully_observable: True if fully observable
            use_mission: True if mission should be in the observation, in which case:
             if using image_based grid, the observation is a dict of keys, image and mission.
             if using flattened observations, then the observation has the mission encoded in it.
        """

        super(MiniGridSingleAgent, self).__init__(env_name=env_name)

        if fully_observable:
            self._env = FullyObsWrapper(self._env)
            if image_based_obs:
                self._env = RGBImgObsWrapper(self._env)
        elif image_based_obs:
            self._env = RGBImgPartialObsWrapper(self._env)

        if use_mission:
            if not image_based_obs:
                self._env = FlatObsWrapper(self._env)
        else:
            if not image_based_obs:
                self._env = FlatObsNoMissionWrapper(self._env)
            else:
                self._env = ImgObsWrapper(self._env)

        self.env_spec = EnvSpec(env_name=env_name,
                                obs_dim=self._env.observation_space.shape,
                                act_dim=self._env.action_space.n)

    def render(self, mode='rgb_array'):
        # TODO
        pass
