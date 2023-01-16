import numpy as np
from hive.debugger.DebuggerInterface import DebuggerInterface
import tensorflow as tf
from hive.debugger.utils.model_params_getters import get_loss, get_model_weights_and_biases
from hive.utils.torch_utils import numpify


class LossCheck(DebuggerInterface):

    def __init__(self, check_period):
        super().__init__()
        self.check_type = "Loss"
        self.check_period = check_period
        self.iter_num = -1

    def run(self, labels, predictions, loss, model):
        error_msgs = list()
        losses = []
        n = self.config["init_loss"]["size_growth_rate"]
        while n <= (self.config["init_loss"]["size_growth_rate"] * self.config["init_loss"]["size_growth_iters"]):
            derived_batch_y = np.concatenate(n * [labels], axis=0)
            derived_predictions = np.concatenate(n * [numpify(predictions)], axis=0)
            loss_value = float(get_loss(derived_predictions, derived_batch_y, loss))
            losses.append(loss_value)
            n *= self.config["init_loss"]["size_growth_rate"]
        rounded_loss_rates = [round(losses[i + 1] / losses[i]) for i in range(len(losses) - 1)]
        equality_checks = sum(
            [(loss_rate == self.config["init_loss"]["size_growth_rate"]) for loss_rate in rounded_loss_rates])
        if equality_checks == len(rounded_loss_rates):
            error_msgs.append(self.main_msgs['poor_reduction_loss'])

        initial_loss = float(get_loss(predictions, labels, loss))
        # specify here the number of actions
        initial_weights, _ = get_model_weights_and_biases(model)
        number_of_actions = list(initial_weights.items())[-1][1].shape[0]
        expected_loss = -np.log(1 / number_of_actions)
        err = np.abs(initial_loss - expected_loss)
        if err >= self.config["init_loss"]["dev_ratio"] * expected_loss:
            error_msgs.append(self.main_msgs['poor_init_loss'].format(round((err / expected_loss), 3)))
        return error_msgs
