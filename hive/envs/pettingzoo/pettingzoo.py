from importlib import import_module

import numpy as np

from hive.envs import BaseEnv
from hive.envs.env_spec import EnvSpec


class PettingZooEnv(BaseEnv):
    """
    PettingZoo environment from https://github.com/PettingZoo-Team/PettingZoo

    For now, we only support environments from PettingZoo with discrete actions.
    """

    def __init__(
        self,
        env_name,
        env_family,
        num_players,
        **kwargs,
    ):
        """
        Args:
            env_name (str): Name of the environment
            env_family (str): Family of the environment such as "Atari",
            "Classic", "SISL", "Butterfly", "MAgent", and "MPE".
            num_players (int): Number of learning agents
        """
        self._env_family = env_family
        self.create_env(env_name, num_players, **kwargs)
        self._env_spec = self.create_env_spec(env_name, **kwargs)
        super().__init__(self.create_env_spec(env_name, **kwargs), num_players)

    def create_env(self, env_name, num_players, **kwargs):
        env_module = import_module("pettingzoo." + self._env_family + "." + env_name)
        self._env = env_module.env(players=num_players)

    def create_env_spec(self, env_name, **kwargs):
        """
        Each family of environments have their own type of observations and actions.
        You can add support for more families here by modifying obs_dim and act_dim.
        """
        if self._env_family in ["classic"]:
            obs_dim = [
                space["observation"].shape
                for space in self._env.observation_spaces.values()
            ]
        elif self._env_family in ["sisl"]:
            obs_dim = [space.shape for space in self._env.observation_spaces.values()]
        else:
            raise ValueError(
                f"Hive does not support {self._env_family} environments from PettingZoo yet."
            )
        act_dim = [space.n for space in self._env.action_spaces.values()]
        return EnvSpec(
            env_name=env_name,
            obs_dim=obs_dim,
            act_dim=act_dim,
        )

    def reset(self):
        self._env.reset()
        observation, _, _, _ = self._env.last()
        for key in observation.keys():
            observation[key] = np.array(observation[key], dtype=np.uint8)
        self._turn = self._env.agents.index(self._env.agent_selection)

        return observation, self._turn

    def step(self, action):
        self._env.step(action)
        observation, _, done, info = self._env.last()
        self._turn = (self._turn + 1) % self._num_players
        for key in observation.keys():
            observation[key] = np.array(observation[key], dtype=np.uint8)
        return (
            observation,
            [self._env.rewards[agent] for agent in self._env.agents],
            done,
            self._turn,
            info,
        )

    def render(self, mode="rgb_array"):
        return self._env.render(mode=mode)

    def seed(self, seed=None):
        self._env.seed(seed=seed)

    def close(self):
        self._env.close()
