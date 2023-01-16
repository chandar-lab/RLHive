import numpy as np

from hive.debugger.DebuggerInterface import DebuggerInterface
from hive.debugger.utils.model_params_getters import get_model_weights_and_biases


class BiasCheck(DebuggerInterface):

    def __init__(self, check_period):
        super().__init__()
        self.check_type = "Bias"
        self.check_period = check_period
        self.iter_num = -1

    def run(self, model):
        error_msg = list()
        _, initial_biases = get_model_weights_and_biases(model)
        if not initial_biases:
            error_msg.append(self.main_msgs['need_bias'])
        else:
            checks = []
            for b_name, b_array in initial_biases.items():
                checks.append(np.sum(b_array) == 0.0)
            if not np.all(checks):
                error_msg.append(self.main_msgs['zero_bias'])
        return error_msg
