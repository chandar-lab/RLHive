import torch.nn
import numpy as np

from hive.debugger.DebuggerInterface import DebuggerInterface
from hive.debugger.utils.metrics import almost_equal
from hive.debugger.utils import metrics
from hive.debugger.utils.model_params_getters import get_model_layer_names, get_model_weights_and_biases


class WeightsCheck(DebuggerInterface):

    def __init__(self, check_period):
        super().__init__()
        self.check_type = "Weight"
        self.check_period = check_period
        self.iter_num = -1

    def run(self, model):
        error_msg = list()
        initial_weights, _ = get_model_weights_and_biases(model)
        layer_names = get_model_layer_names(model)
        self.iter_num += 1
        for layer_name, weight_array in initial_weights.items():
            shape = weight_array.shape
            if len(shape) == 1 and shape[0] == 1:
                continue
            if almost_equal(np.var(weight_array), 0.0, rtol=1e-8):
                error_msg.append(self.main_msgs['poor_init'].format(layer_name))
            else:
                if len(shape) == 2:
                    fan_in = shape[0]
                    fan_out = shape[1]
                else:
                    receptive_field_size = np.prod(shape[:-2])
                    fan_in = shape[-2] * receptive_field_size
                    fan_out = shape[-1] * receptive_field_size
                lecun_F, lecun_test = metrics.pure_f_test(weight_array, np.sqrt(1.0 / fan_in),
                                                          self.config["Initial_Weight"]["f_test_alpha"])
                he_F, he_test = metrics.pure_f_test(weight_array, np.sqrt(2.0 / fan_in),
                                                    self.config["Initial_Weight"]["f_test_alpha"])
                glorot_F, glorot_test = metrics.pure_f_test(weight_array, np.sqrt(2.0 / (fan_in + fan_out)),
                                                            self.config["Initial_Weight"]["f_test_alpha"])

                # The following checks can't be done on the last layer
                if layer_name == next(reversed(layer_names.items()))[0]:
                    break
                activation_layer = list(layer_names)[list(layer_names.keys()).index(layer_name) + 1]

                if isinstance(layer_names[activation_layer], torch.nn.ReLU) and not he_test:
                    abs_std_err = np.abs(np.std(weight_array) - np.sqrt(1.0 / fan_in))
                    error_msg.append(self.main_msgs['need_he'].format(layer_name, abs_std_err))
                elif isinstance(layer_names[activation_layer], torch.nn.Tanh) and not glorot_test:
                    abs_std_err = np.abs(np.std(weight_array) - np.sqrt(2.0 / fan_in))
                    error_msg.append(self.main_msgs['need_glorot'].format(layer_name, abs_std_err))
                elif isinstance(layer_names[activation_layer], torch.nn.Sigmoid) and not lecun_test:
                    abs_std_err = np.abs(np.std(weight_array) - np.sqrt(2.0 / (fan_in + fan_out)))
                    error_msg.append(self.main_msgs['need_lecun'].format(layer_name, abs_std_err))
                elif not (lecun_test or he_test or glorot_test):
                    error_msg.append(self.main_msgs['need_init_well'].format(layer_name))

        return error_msg
