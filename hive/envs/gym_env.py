import gym

from hive.envs.base import BaseEnv
from hive.envs.env_spec import EnvSpec


class GymEnv(BaseEnv):

    def __init__(self, env_name):
        super(GymEnv, self).__init__()

        self._env = gym.make(env_name)
        self.env_specs = EnvSpec(env_name=env_name,
                                 obs_dim=self._env.observation_space.shape,
                                 act_dim=self._env.action_space.n)
        self._turn = 0

    def reset(self):
        observation = self._env.reset()
        return observation, self._turn

    def step(self, action=None):
        observation, reward, done, info = self._env.step(action)
        return observation, reward, done, self._turn, info

    def render(self, mode='rgb_array'):
        self._env.render(mode=mode)

    def seed(self, seed=None):
        self._env.seed(seed=seed)

    def close(self):
        self._env.close()
