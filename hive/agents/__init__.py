from hive.agents.agent import Agent
from hive.agents.dqn import DQNAgent
from hive.agents.random import RandomAgent
from hive.agents.hanabi_rainbow import HanabiRainbowAgent
from hive.utils.utils import create_class_constructor

get_agent = create_class_constructor(
    Agent,
    {
        "DQNAgent": DQNAgent,
        "RandomAgent": RandomAgent,
        "HanabiRainbowAgent": HanabiRainbowAgent,
    },
)
