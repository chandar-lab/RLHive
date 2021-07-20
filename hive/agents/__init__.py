from hive import registry
from hive.agents.agent import Agent
from hive.agents.reinforce import REINFORCEAgent
from hive.agents.dqn import DQNAgent
from hive.agents.rainbow import RainbowDQNAgent
from hive.agents.random import RandomAgent

registry.register_all(
    Agent,
    {   
        "REINFORCEAgent": REINFORCEAgent,
        "DQNAgent": DQNAgent,
        "RandomAgent": RandomAgent,
        "RainbowDQNAgent": RainbowDQNAgent,
    },
)

get_agent = getattr(registry, f"get_{Agent.type_name()}")
