from hive.debugger.Checkers.NN_checkers.BiasCheck import BiasCheck
from hive.debugger.Checkers.NN_checkers.LossCheck import LossCheck
from hive.debugger.Checkers.NN_checkers.ObservationsCheck import ObservationsCheck
from hive.debugger.Checkers.NN_checkers.ProperFittingCheck import ProperFittingCheck
from hive.debugger.Checkers.NN_checkers.WeightsCheck import WeightsCheck
from hive.debugger.DebuggerInterface import DebuggerInterface
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

registry.register("Observations", ObservationsCheck, ObservationsCheck)
registry.register("Weights", WeightsCheck, WeightsCheck)
registry.register("Bias", BiasCheck, BiasCheck)
registry.register("Loss", LossCheck, LossCheck)
registry.register("ProperFitting", ProperFittingCheck, ProperFittingCheck)



get_debugger = getattr(registry, f"get_{DebuggerInterface.type_name()}")
