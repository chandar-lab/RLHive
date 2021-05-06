from marlgrid.envs import register_marl_env
from hive.envs.ma_envs.checkers import CheckersMultiGrid
from hive.envs.ma_envs.pursuit import PursuitMultiGrid


register_marl_env(
    "MarlGrid-2AgentCheckers8x8-v0",
    CheckersMultiGrid,
    n_agents=2,
    grid_size=8,
    view_size=5,
    env_kwargs={"max_steps": 100},
)

register_marl_env(
    "MarlGrid-2Agent1RandomPursuit8x8-v0",
    PursuitMultiGrid,
    n_agents=3,
    grid_size=8,
    view_size=5,
    env_kwargs={"max_steps": 500},
)
