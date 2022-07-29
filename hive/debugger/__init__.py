from hive.utils.registry import registry
from hive.debugger.debugger import Debugger

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

get_debugger = getattr(registry, f"get_{Debugger.type_name()}")