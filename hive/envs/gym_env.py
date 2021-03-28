import gym

from hive.envs.base import BaseEnv
from hive.envs.env_spec import EnvSpec


class GymEnv(BaseEnv):
    """
    Class for loading gym built-in environments.
    """

    def __init__(self, env_name, num_players, **kwargs):
        """
        Args:
            env_name: Name of the environment (NOTE: make sure it is available at gym.envs.registry.all())
        """
        self.create_env(env_name, **kwargs)
        super().__init__(self.create_env_spec(env_name, **kwargs), num_players)

    def create_env(self, env_name, **kwargs):
        self._env = gym.make(env_name)

    def create_env_spec(self, env_name, **kwargs):
        if isinstance(self._env.observation_space, gym.spaces.Tuple):
            obs_spaces = self._env.observation_space.spaces
        else:
            obs_spaces = [self._env.observation_space]
        if isinstance(self._env.action_space, gym.spaces.Tuple):
            act_spaces = self._env.action_space.spaces
        else:
            act_spaces = [self._env.action_space]

        return EnvSpec(
            env_name=env_name,
            obs_dim=[space.shape for space in obs_spaces],
            act_dim=[space.n for space in act_spaces],
        )

    def reset(self):
        observation = self._env.reset()
        return observation, self._turn

    def step(self, action=None):
        observation, reward, done, info = self._env.step(action)
        self._turn = (self._turn + 1) % self._num_players
        return observation, reward, done, self._turn, info

    def render(self, mode="rgb_array"):
        return self._env.render(mode=mode)

    def seed(self, seed=None):
        self._env.seed(seed=seed)

    def close(self):
        self._env.close()
