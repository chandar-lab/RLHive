import numpy as np
import gym
import torch

from hive.envs.gym_env import GymEnv
from hive.envs.env_spec import EnvSpec
from importlib import import_module


class MinAtarEnv():
    """
    Class for loading Atari environments.
    """

    def __init__(
        self,
        env_name,
        sticky_action_prob=0.1,
        difficulty_ramping=True,
        random_seed=None,
    ):
        """
        Args:
            env_name (str): Name of the environment
            sticky_actions (boolean): Whether to use sticky_actions as per Machado et al.
        """
        env_module = import_module("minatar.environments." + env_name)
        self.env_name = env_name
        self._env = env_module.Env(ramping=difficulty_ramping)
        # self._env = self.create_env(self.env_name)
        self.n_channels = self._env.state_shape()[2]
        self.sticky_action_prob = sticky_action_prob
        self.last_action = 0
        self.visualized = False
        self.closed = False
        self.env_spec = self.create_env_spec(env_name)

    def create_env_spec(self, env_name, **kwargs):
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
        return (
            torch.tensor(self._env.state()).permute(2, 0, 1)
        ).float().detach().cpu().numpy(), 0

    def step(self, action=None):
        """
        Remarks:
            * Execute self.frame_skips steps taking the action in the the environment.
            * This may execute fewer than self.frame_skip steps in the environment, if the done state is reached.
            * Furthermore, in this case the returned observation should be ignored.
        """
        assert action is not None
        reward, done = self._env.act(action)
        reward = float(reward)
        info = {}
        observation = (
            torch.tensor(self._env.state())
            .permute(2, 0, 1)
            .float()
            .detach()
            .cpu()
            .numpy()
        )

        return observation, reward, done, None, info
