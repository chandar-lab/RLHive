from hive.utils.registry import registry
from hive.runners.base import Runner
from hive.runners.single_agent_loop import SingleAgentRunner
from hive.runners.multi_agent_loop import MultiAgentRunner

registry.register_all(
    Runner,
    {
        "SingleAgentRunner": SingleAgentRunner,
        "MultiAgentRunner": MultiAgentRunner,
    },
)

get_runner = getattr(registry, f"get_{Runner.type_name()}")
