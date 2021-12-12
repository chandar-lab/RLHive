from importlib import import_module

import numpy as np

from hive.envs.base import BaseEnv
from hive.envs.env_spec import EnvSpec


class MinAtarEnv(BaseEnv):
    """
    Class for loading MinAtar environments. See https://github.com/kenjyoung/MinAtar.
    """

    def __init__(
        self,
        env_name,
        sticky_action_prob=0.1,
        difficulty_ramping=True,
    ):
        """
        Args:
            env_name (str): Name of the environment
            sticky_actions (bool): Whether to use sticky_actions as per
                Machado et al.
            difficulty_ramping (bool): Whether to periodically increase difficulty.
        """
        env_module = import_module("minatar.environments." + env_name)
        self.env_name = env_name
        self._env = env_module.Env(ramping=difficulty_ramping)
        self.n_channels = self._env.state_shape()[2]
        self.sticky_action_prob = sticky_action_prob
        self.last_action = 0
        self.visualized = False
        self.closed = False
        super().__init__(self.create_env_spec(env_name), num_players=1)

    def create_env_spec(self, env_name):
        obs_dim = tuple(self._env.state_shape())
        new_positions = [2, 0, 1]
        obs_dim = tuple(obs_dim[i] for i in new_positions)
        return EnvSpec(
            env_name=env_name,
            obs_dim=[obs_dim],
            act_dim=[6],
        )

    def reset(self):
        self._env.reset()
        return np.transpose(self._env.state(), [2, 1, 0]), 0

    def seed(self, seed=None):
        self._env.seed(seed=seed)

    def step(self, action=None):
        """
        Remarks:
            * Execute self.frame_skips steps taking the action in the the environment.
            * This may execute fewer than self.frame_skip steps in the environment,
              if the done state is reached.
            * Furthermore, in this case the returned observation should be ignored.
        """
        assert action is not None
        reward, done = self._env.act(action)
        reward = float(reward)
        info = {}
        observation = np.transpose(self._env.state(), [2, 1, 0])

        return observation, reward, done, None, info
