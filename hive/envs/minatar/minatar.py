import numpy as np
import gym
import torch

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
        env_module = import_module('hive.envs.minatar.environments.' + env_name)
        self.env_name = env_name
        # gym.envs.register(
        #     id=self.env_name,
        #     entry_point='hive.envs.minatar.minatar:MinAtarEnv',
        #     max_episode_steps=1000,
        # )
        self._env = env_module.Env(ramping=difficulty_ramping, seed=random_seed)
        self.n_channels = self._env.state_shape()[2]
        self.sticky_action_prob = sticky_action_prob
        self.last_action = 0
        self.visualized = False
        self.closed = False
        self.env_spec = self.create_env_spec(env_name)
        # super().__init__(env_name=self.env_name)

    def create_env_spec(self, env_name, **kwargs):
        act_spaces = [6]
        print("obs_dim = ", self._env.state_shape())
        obs_dim = tuple(self._env.state_shape())
        new_positions = [2, 0, 1]
        print("obs dim[2] = ", obs_dim[0])
        obs_dim = tuple(obs_dim[i] for i in new_positions)
        print("obs_dim = ", obs_dim)
        return EnvSpec(
            env_name=env_name,
            # obs_dim=[(1, 10, 10)],
            obs_dim=[obs_dim],
            act_dim=[6],
        )

    def reset(self):
        self._env.reset()
        return (torch.tensor(self._env.state()).permute(2, 0, 1)).float(), 0

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
        observation = torch.tensor(self._env.state()).permute(2, 0, 1).float()
        self._turn = 0
        print("inside step")
        print("observation = ", observation)
        print("reward = ", reward)
        print("done = ", done)
        print("info = ", info)

        return observation.cpu().detach().numpy(), reward, done, self._turn, info
