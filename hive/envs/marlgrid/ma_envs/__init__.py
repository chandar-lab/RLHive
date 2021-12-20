from gym.envs.registration import register
from marlgrid.envs import register_marl_env

from hive.envs.marlgrid.ma_envs.checkers import CheckersMultiGrid
from hive.envs.marlgrid.ma_envs.pursuit import PursuitMultiGrid
from hive.envs.marlgrid.ma_envs.switch import SwitchMultiGrid

register_marl_env(
    "MarlGrid-2AgentCheckers8x8-v0",
    CheckersMultiGrid,
    n_agents=2,
    grid_size=8,
    view_size=5,  # Needs to be same as grid_size if full_obs = True.
    env_kwargs={"max_steps": 100, "full_obs": False},
)

register_marl_env(
    "MarlGrid-2Agent1RandomPursuit8x8-v0",
    PursuitMultiGrid,
    n_agents=3,
    grid_size=8,
    view_size=5,  # Needs to be same as grid_size if full_obs = True.
    env_kwargs={"max_steps": 500, "full_obs": False},
)

register_marl_env(
    "MarlGrid-2AgentSwitch8x8-v0",
    SwitchMultiGrid,
    n_agents=2,
    grid_size=12,
    view_size=12,  # Needs to be same as grid_size if full_obs = True.
    env_kwargs={"max_steps": 500, "full_obs": True},
)
