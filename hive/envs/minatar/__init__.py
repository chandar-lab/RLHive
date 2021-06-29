from hive.envs.minatar.minatar import MinAtarEnv

from gym.envs.registration import register

register(
    id="breakout-v0",
    entry_point="hive.envs.minatar.minatar:MinAtarEnv",
)
