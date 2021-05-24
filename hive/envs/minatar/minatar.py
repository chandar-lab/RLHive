import numpy as np
import cv2

from hive.envs.gym_env import GymEnv
from hive.envs.env_spec import EnvSpec


class MinAtarEnv(GymEnv):
    """
    Class for loading Atari environments.
    """

    def __init__(
        self,
        env_name,
        sticky_actions=True,
    ):
        """
        Args:
            env_name (str): Name of the environment
            sticky_actions (boolean): Whether to use sticky_actions as per Machado et al.
        """
        super().__init__(env_name)

    def create_env_spec(self, env_name, **kwargs):
        obs_spaces = self._env.observation_space.shape
        # Used for storing and pooling over two consecutive observations
        act_spaces = [self._env.action_space]
        return EnvSpec(
            env_name=env_name,
            obs_dim=[(1, 10, 10)],
            act_dim=[space.n for space in act_spaces],
        )

    def reset(self):
        self._env.reset()
        return self._pool_and_resize(), self._turn

    def step(self, action=None):
        """
        Remarks:
            * Execute self.frame_skips steps taking the action in the the environment.
            * This may execute fewer than self.frame_skip steps in the environment, if the done state is reached.
            * Furthermore, in this case the returned observation should be ignored.
        """
        assert action is not None
        observation, reward, done, info = self._env.act(action)

        return observation, reward, done, self._turn, info
