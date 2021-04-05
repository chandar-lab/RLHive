from hive.agents.agent import Agent
from hive.agents.dqn import DQNAgent
from hive.agents.double_dqn import DoubleDQNAgent
from hive.agents.dueling_dqn import DuelingDQNAgent
from hive.agents.noisy_dqn import NoisyDQNAgent
from hive.utils.utils import create_class_constructor

get_agent = create_class_constructor(Agent, {"DQNAgent": DQNAgent,
                                             "DoubleDQNAgent": DoubleDQNAgent,
                                             "DuelingDQNAgent": DuelingDQNAgent,
                                             "NoisyDQNAgent": NoisyDQNAgent
                                             }
                                     )
