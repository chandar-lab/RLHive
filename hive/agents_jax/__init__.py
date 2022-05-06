from hive.agents_jax import qnets
from hive.agents.agent import Agent
from hive.agents_jax.jax_dqn import JaxDQNAgent
from hive.utils.registry import registry

registry.register_all(
    Agent, {"JaxDQNAgent": JaxDQNAgent,},
)

get_agent = getattr(registry, f"get_{Agent.type_name()}")
