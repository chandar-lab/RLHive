import gym

from hive.envs.base import BaseEnv
from hive.envs.env_spec import EnvSpec


class GymEnv(BaseEnv):
    """
    Class for loading gym built-in environments.
    """

    def __init__(self, env_name):
        """
        Args:
            env_name: Name of the environment (NOTE: make sure it is available at gym.envs.registry.all())
        """
        super(GymEnv, self).__init__()

        self._env = gym.make(env_name)
        self.env_spec = EnvSpec(env_name=env_name,
                                obs_dim=self._env.observation_space.shape,
                                act_dim=self._env.action_space.n)
        
        # Since we have a single agent environment, the only agent should take a turn at each time step.
        # So, turn is always zero.
        self._turn = 0

    def reset(self):
        observation = self._env.reset()
        return observation, self._turn

    def step(self, action=None):
        observation, reward, done, info = self._env.step(action)
        return observation, reward, done, self._turn, info

    def render(self, mode='rgb_array'):
        return self._env.render(mode=mode)

    def seed(self, seed=None):
        self._env.seed(seed=seed)

    def close(self):
        self._env.close()
