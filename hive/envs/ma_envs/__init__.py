from marlgrid.envs import register_marl_env
from hive.envs.ma_envs.checkers import CheckersMultiGrid

register_marl_env(
    "MarlGrid-2AgentCheckers8x8-v0",
    CheckersMultiGrid,
    n_agents=2,
    grid_size=8,
    view_size=5,
    env_kwargs={}
)