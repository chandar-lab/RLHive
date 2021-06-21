"""Based on Probe Environments from https://andyljones.com/posts/rl-debugging.html"""
from hive.envs.base import BaseEnv
from hive.envs.env_spec import EnvSpec
import numpy as np
import random


class BaseProbeEnv(BaseEnv):
    def __init__(self, env_name, img, act_dim):
        self.img = img
        env_spec = EnvSpec(
            env_name, obs_dim=[(1, 5, 5)] if img else [(5,)], act_dim=[act_dim]
        )
        super().__init__(env_spec, num_players=1)

    def create_observation(self):
        if self.img:
            return np.ones((1, 5, 5))
        else:
            return np.ones(5)

    def reset(self):
        return self.create_observation(), 0

    def step(self, action):
        raise NotImplementedError()

    def seed(self, seed):
        pass


class Probe1(BaseProbeEnv):
    def __init__(self, img=False):
        super().__init__("Probe1", img, act_dim=1)

    def step(self, action):
        action = int(action)
        if action != 0:
            raise ValueError("Invalid Action")
        return self.create_observation(), 1, True, 0, {}


class Probe2(BaseProbeEnv):
    def __init__(self, img=False):
        super().__init__("Probe2", img, act_dim=1)
        self.prev_obs = None

    def create_observation(self):
        obs = super().create_observation()
        if random.random() < 0.5:
            return obs * -1
        else:
            return obs

    def reset(self):
        self.prev_obs = self.create_observation()
        return self.prev_obs, 0

    def step(self, action):
        action = int(action)
        if action != 0:
            raise ValueError("Invalid Action")
        reward = int(self.prev_obs.flatten()[0])
        self.prev_obs = None
        return self.create_observation(), reward, True, 0, {}

    def seed(self, seed):
        random.seed(seed)


class Probe3(BaseProbeEnv):
    def __init__(self, img=False):
        super().__init__("Probe3", img, act_dim=1)
        self.step_num = 0

    def create_observation(self):
        obs = super().create_observation()
        return obs * self.step_num

    def reset(self):
        self.step_num = 0
        return self.create_observation(), 0

    def step(self, action):
        action = int(action)
        if action != 0:
            raise ValueError("Invalid Action")
        self.step_num += 1
        if self.step_num == 2:
            return self.create_observation(), 1, True, 0, {}
        else:
            return self.create_observation(), 0, False, 0, {}


class Probe4(BaseProbeEnv):
    def __init__(self, img=False):
        super().__init__("Probe4", img, act_dim=2)

    def step(self, action):
        action = int(action)
        if action not in [0, 1]:
            raise ValueError("Invalid Action")
        reward = 1 if action == 1 else -1
        return self.create_observation(), reward, True, 0, {}


class Probe5(BaseProbeEnv):
    def __init__(self, img=False):
        super().__init__("Probe5", img, act_dim=2)
        self.prev_obs = None

    def create_observation(self):
        obs = super().create_observation()
        if random.random() < 0.5:
            return obs * -1
        else:
            return obs

    def reset(self):
        self.prev_obs = self.create_observation()
        return self.prev_obs, 0

    def step(self, action):
        action = int(action)
        if action not in [0, 1]:
            raise ValueError("Invalid Action")
        obs = int((self.prev_obs.flatten()[0] + 1) / 2)
        self.prev_obs = None
        reward = 1 if action == obs else -1
        return self.create_observation(), reward, True, 0, {}

    def seed(self, seed):
        random.seed(seed)
