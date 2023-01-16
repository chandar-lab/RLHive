import copy

import numpy as np
import torch.nn

from hive.debugger.DebuggerInterface import DebuggerInterface
from hive.debugger.utils.metrics import smoothness, are_significantly_different
from hive.debugger.utils.model_params_getters import get_loss, get_model_weights_and_biases


class ProperFittingCheck(DebuggerInterface):

    def __init__(self, check_period):
        super().__init__()
        self.check_type = "ProperFitting"
        self.check_period = check_period
        self.iter_num = -1

    def run(self, observations, labels, actions, opt, model, loss):
        def _loss_is_stable(loss_value):
            if np.isnan(loss_value):
                return self.main_msgs['nan_loss']
            if np.isinf(loss_value):
                return self.main_msgs['inf_loss']
            return True

        opt = opt.__class__(model.parameters(),)
        error_msg = list()
        derived_batch_x = observations
        derived_batch_y = labels

        real_losses = []
        model.train(True)
        for i in range(self.config["total_iters"]):
            print(i)
            opt.zero_grad()
            outputs = model(torch.tensor(derived_batch_x))
            outputs = outputs[torch.arange(outputs.size(0)), actions]
            loss_value = get_loss(outputs, derived_batch_y, loss)
            loss_value.backward()
            opt.step()
            real_losses.append(loss_value.item())
            if not (_loss_is_stable(loss_value.item())):
                error_msg.append(self.main_msgs['underfitting_single_batch'])


        # loss_smoothness = smoothness(np.array(real_losses))
        # min_loss = np.min(np.array(real_losses))
        # if min_loss <= self.config.prop_fit.abs_loss_min_thresh or (
        #         min_loss <= self.config.prop_fit.loss_min_thresh and
        #         loss_smoothness > self.config.prop_fit.smoothness_max_thresh):
        #     return self.main_msgs['zero_loss']
        #
        # zeroed_batch_x = np.zeros_like(derived_batch_x)
        # fake_losses = []
        # for _ in range(self.config.prop_fit.total_iters):
        #     outputs = model(torch.tensor(zeroed_batch_x))
        #     outputs = outputs[torch.arange(outputs.size(0)), actions]
        #     fake_loss = float(get_loss(outputs, derived_batch_y, loss))
        #     fake_losses.append(fake_loss)
        #     if not (_loss_is_stable(fake_loss)):
        #         return ""
        # stability_test = np.array([_loss_is_stable(loss_value) for loss_value in (real_losses + fake_losses)])
        # if (stability_test == False).any():
        #     last_real_losses = real_losses[-self.config.prop_fit.sample_size_of_losses:]
        #     last_fake_losses = fake_losses[-self.config.prop_fit.sample_size_of_losses:]
        #     if not (are_significantly_different(last_real_losses, last_fake_losses)):
        #         return self.main_msgs['data_dep']

        return error_msg


        # weights1, _ = get_model_weights_and_biases(model)
        # weights2, _ = get_model_weights_and_biases(modelllll)
        #
        # for k in weights1.keys():
        #     print((weights1[k] == weights2[k]).all())