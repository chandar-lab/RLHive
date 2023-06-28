from hive.agents import networks
from hive.agents.agent import Agent
from hive.agents.dqn import DQNAgent, LegalMovesRainbowAgent, RainbowDQNAgent
from hive.agents.ppo.ppo import PPOAgent
from hive.agents.random import RandomAgent
from hive.agents.sac import DiscreteSACAgent, SACAgent
from hive.agents.sequence.drqn import DRQNAgent
from hive.agents.td3 import DDPG, TD3
from hive.utils.registry import registry

registry.register_classes(
    {
        "DDPG": DDPG,
        "DiscreteSACAgent": DiscreteSACAgent,
        "DQNAgent": DQNAgent,
        "DRQNAgent": DRQNAgent,
        "LegalMovesRainbowAgent": LegalMovesRainbowAgent,
        "PPOAgent": PPOAgent,
        "RainbowDQNAgent": RainbowDQNAgent,
        "RandomAgent": RandomAgent,
        "SACAgent": SACAgent,
        "TD3": TD3,
    },
)
