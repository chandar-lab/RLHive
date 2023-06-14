from hive.runners.base import Runner
from hive.runners.multi_agent_loop import MultiAgentRunner

# from hive.runners.parallel_single_agent_loop import ParallelSingleAgentRunner
from hive.runners.single_agent_loop import SingleAgentRunner
from hive.utils.registry import registry

registry.register_classes(
    {
        "SingleAgentRunner": SingleAgentRunner,
        "MultiAgentRunner": MultiAgentRunner,
        # "ParallelSingleAgentRunner": ParallelSingleAgentRunner,
    },
)
