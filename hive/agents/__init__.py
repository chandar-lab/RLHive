import logging

from hive.agents import qnets
from hive.agents.agent import Agent
from hive.agents.ddpg import DDPG
from hive.agents.dqn import DQNAgent
from hive.agents.drqn import DRQNAgent
from hive.agents.legal_moves_rainbow import LegalMovesRainbowAgent
from hive.agents.rainbow import RainbowDQNAgent
from hive.agents.random import RandomAgent
from hive.agents.td3 import TD3
from hive.utils.registry import registry

registry.register_all(
    Agent,
    {
        "DDPG": DDPG,
        "DQNAgent": DQNAgent,
        "DRQNAgent": DRQNAgent,
        "LegalMovesRainbowAgent": LegalMovesRainbowAgent,
        "RainbowDQNAgent": RainbowDQNAgent,
        "RandomAgent": RandomAgent,
        "TD3": TD3,
    },
)

logging.info("Registered agents.")

get_agent = getattr(registry, f"get_{Agent.type_name()}")
