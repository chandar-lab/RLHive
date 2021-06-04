from hive.agents.agent import Agent
from hive.agents.dqn import DQNAgent
from hive.agents.hanabi_dqn import HanabiDQNAgent
from hive.agents.random import RandomAgent
from hive.utils.utils import create_class_constructor

get_agent = create_class_constructor(
    Agent,
    {
        "DQNAgent": DQNAgent,
        "RandomAgent": RandomAgent,
        "HanabiDQNAgent": HanabiDQNAgent,
    },
)
