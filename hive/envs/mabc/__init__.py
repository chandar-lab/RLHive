from gym.envs.registration import register

register(
    id="MABC-v0",
    entry_point="hive.envs.mabc.MABCEnv:MABC",
    max_episode_steps=300,
)
