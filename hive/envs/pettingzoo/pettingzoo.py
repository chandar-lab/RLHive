import numpy as np

from hive.envs import ParallelEnv, GymEnv
from importlib import import_module
from hive.envs.env_spec import EnvSpec


class PettingZooEnv(ParallelEnv, GymEnv):
    """
    PettingZoo environemnt from https://github.com/PettingZoo-Team/PettingZoo

    """

    def __init__(
        self,
        env_name,
        env_family,
        num_players,
    ):
        self._env_family = env_family
        self.create_env(env_name)
        self._env_spec = self.create_env_spec(env_name)
        super().__init__(env_name, num_players)

    def create_env(self, env_name, **kwargs):

        env_module = import_module("pettingzoo." + self._env_family + "." + env_name)
        self._env = env_module.env()

    def create_env_spec(self, env_name, **kwargs):
        if self._env_family in ["classic"]:
            obs_dim = [
                space["observation"].shape
                for space in self._env.observation_spaces.values()
            ]
        elif self._env_family in ["sisl"]:
            obs_dim = [space.shape for space in self._env.observation_spaces.values()]
        act_dim = [space.n for space in self._env.action_spaces.values()]
        return EnvSpec(
            env_name=env_name,
            obs_dim=obs_dim,
            act_dim=act_dim,
        )

    def reset(self):
        self._env.reset()
        observation, _, _, _ = self._env.last()

        return observation, self._turn

    def step(self, action):

        self._env.step(action)
        observation, reward, done, info = self._env.last()

        self._done = done
        self._turn = (self._turn + 1) % self._num_players
        self._info = info
        return (
            observation,
            reward,
            self._done,
            self._turn,
            self._info,
        )
