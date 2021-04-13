from hive.agents.agent import Agent
from hive.agents.dqn import DQNAgent
from hive.agents.rainbow import RainbowDQNAgent
from hive.utils.utils import create_class_constructor

get_agent = create_class_constructor(Agent, {"DQNAgent": DQNAgent,
                                             "RainbowDQNAgent": RainbowDQNAgent
                                             })
