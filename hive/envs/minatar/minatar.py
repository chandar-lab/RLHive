import numpy as np

from hive.envs.gym_env import GymEnv
from hive.envs.env_spec import EnvSpec
from importlib import import_module


class MinAtarEnv(GymEnv):
    """
    Class for loading Atari environments.
    """

    def __init__(
        self,
        env_name,
        sticky_action_prob=0.1,
        difficulty_ramping=True,
        random_seed=None
    ):
        """
        Args:
            env_name (str): Name of the environment
            sticky_actions (boolean): Whether to use sticky_actions as per Machado et al.
        """
        print("MinAtar env called")
        env_module = import_module('minatar.environments.' + env_name)
        self.env_name = env_name
        self.env = env_module.Env(ramping=difficulty_ramping, seed=random_seed)
        self.n_channels = self.env.state_shape()[2]
        self.sticky_action_prob = sticky_action_prob
        self.last_action = 0
        self.visualized = False
        self.closed = False
        super().__init__(self.env_name)

    def create_env_spec(self, env_name, **kwargs):
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
