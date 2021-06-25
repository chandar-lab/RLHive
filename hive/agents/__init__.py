from hive import registry
from hive.agents.agent import Agent
from hive.agents.dqn import DQNAgent
from hive.agents.rainbow import RainbowDQNAgent
from hive.agents.random import RandomAgent

registry.register_all(
    Agent,
    {
        "DQNAgent": DQNAgent,
        "RandomAgent": RandomAgent,
        "RainbowDQNAgent": RainbowDQNAgent,
    },
)

get_agent = getattr(registry, f"get_{Agent.type_name()}")
