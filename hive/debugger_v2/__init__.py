from hive.debugger_v2.debugger import Debugger, ObservationsCheck, WeightsCheck
from hive.utils.registry import registry

# registry.register_all(
#     Debugger,
#     {
#         "NullDebugger": NullDebugger,
#         "PreCheckDebugger": PreCheckDebugger,
#         "PostCheckDebugger": PostCheckDebugger,
#         "OnTrainingCheckDebugger": OnTrainingCheckDebugger,
#         "CompositeDebugger": CompositeDebugger,
#     },
# )

registry.register("Debugger", Debugger, Debugger)
registry.register("Observations", ObservationsCheck, ObservationsCheck)
registry.register("Weights", WeightsCheck, WeightsCheck)


get_debugger = getattr(registry, f"get_{Debugger.type_name()}")