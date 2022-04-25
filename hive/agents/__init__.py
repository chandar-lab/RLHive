from hive.agents import qnets
from hive.agents.agent import Agent
from hive.agents.dqn import DQNAgent
from hive.agents.drqn import DRQNAgent
from hive.agents.legal_moves_rainbow import LegalMovesRainbowAgent
from hive.agents.rainbow import RainbowDQNAgent
from hive.agents.random import RandomAgent
from hive.agents.drqn_hidden import DRQNhiddenAgent
from hive.utils.registry import registry

registry.register_all(
    Agent,
    {
        "DQNAgent": DQNAgent,
        "LegalMovesRainbowAgent": LegalMovesRainbowAgent,
        "RainbowDQNAgent": RainbowDQNAgent,
        "RandomAgent": RandomAgent,
        "DRQNAgent": DRQNAgent,
        "DRQNhiddenAgent": DRQNhiddenAgent,
    },
)

get_agent = getattr(registry, f"get_{Agent.type_name()}")
