from hive.utils.registry import registry
from hive.runners.base import Runner
from hive.runners.single_agent_loop import SingleAgentRunner
from hive.runners.multi_agent_loop import MultiAgentRunner
from hive.runners.parallel_single_agent_loop import ParallelSingleAgentRunner

registry.register_all(
    Runner,
    {
        "SingleAgentRunner": SingleAgentRunner,
        "MultiAgentRunner": MultiAgentRunner,
        "ParallelSingleAgentRunner": ParallelSingleAgentRunner,
    },
)

get_runner = getattr(registry, f"get_{Runner.type_name()}")
