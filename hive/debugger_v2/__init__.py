from hive.debugger_v2.Checkers.ObservationsCheck import ObservationsCheck
from hive.debugger_v2.Checkers.WeightsCheck import WeightsCheck
from hive.debugger_v2.DebuggerInterface import DebuggerInterface
from hive.utils.registry import registry

# Todo
#  registry.register_all(
#     Debugger,
#     {
#         "NullDebugger": NullDebugger,
#         "PreCheckDebugger": PreCheckDebugger,
#         "PostCheckDebugger": PostCheckDebugger,
#         "OnTrainingCheckDebugger": OnTrainingCheckDebugger,
#         "CompositeDebugger": CompositeDebugger,
#     },
# )

registry.register("Debugger", DebuggerInterface, DebuggerInterface)
registry.register("Observations", ObservationsCheck, ObservationsCheck)
registry.register("Weights", WeightsCheck, WeightsCheck)


get_debugger = getattr(registry, f"get_{DebuggerInterface.type_name()}")