from pathlib import Path
import torch.nn
import numpy as np
from hive import debugger_v2 as debugger_lib
from hive.debugger_v2.utils.model_params_getters import get_model_weights_and_biases, get_model_layer_names
from hive.utils.registry import Registrable
from hive.debugger_v2.utils import settings
from hive.debugger_v2.utils import metrics


def almost_equal(value1, value2, rtol=1e-2):
    rerr = np.abs(value1 - value2)
    if isinstance(value1, np.ndarray):
        return (rerr <= rtol).all()
    else:
        return rerr <= rtol


class Debugger(Registrable):
    def __init__(self, logger='log.txt', config='utils/config/settings.yaml', app_path=None):
        # TODO: merge this logger with RLHive's logger
        app_path = Path.cwd() if app_path == None else app_path
        log_fpath = settings.build_log_file_path(app_path, logger)
        self.logger = settings.file_logger(log_fpath, logger)
        # TODO: we need to follow the same config logic as RLHive !!!!
        config_fpath = settings.load_user_config_if_exists(app_path)
        self.config = settings.Config(config_fpath).full_conf
        self.main_msgs = settings.load_messages()
        self.main_msgs = settings.load_messages()
        self.debuggers = dict()

        self.observations = None
        self.model = None
        self.labels = None
        self.predictions = None
        self.loss = None
        self.opt = None
        self.actions = None

    def set_debugger(self, config):
        for debugger_config in config:
            debugger_fn, _ = debugger_lib.get_debugger(debugger_config, debugger_config["name"])
            debugger = debugger_fn(self.config[debugger_config["name"]]["Period"])
            debugger.config = self.config[debugger_config["name"]]
            self.debuggers[debugger_config["name"]] = debugger

    @classmethod
    def type_name(cls):
        return "debugger"

    def set_parameters(self, observations, model, labels, predictions, loss, opt, actions):
        self.observations = observations
        self.model = model
        self.labels = labels
        self.predictions = predictions
        self.loss = loss
        self.opt = opt
        self.actions = actions

    def react(self, message, fail_on=False):
        if fail_on:
            self.main_logger.error(message)
            raise Exception(message)
        else:
            self.logger.warning(message)

    def run(self):
        for debugger in self.debuggers.values():
            if ((debugger.period != 0) and (debugger.actual_period % debugger.period)) \
                    or ((debugger.period == 0) and (debugger.actual_period == -1)) :
                debugger.set_parameters(self.observations, self.model, self.labels, self.predictions,
                                        self.loss, self.opt, self.actions)
                debugger.run()


class ObservationsCheck(Debugger):

    def __init__(self, period):
        super().__init__(logger="Observation_Log")
        self.period = period
        self.actual_period = -1

    def run(self):
        mas = np.max(self.observations)
        mis = np.min(self.observations)
        avgs = np.mean(self.observations)
        stds = np.std(self.observations)

        # for idx in range(len(mas)):
        if stds == 0.0:
            msg = self.main_msgs['features_constant']
            self.react(msg)
        elif any([almost_equal(mas, data_max) for data_max in self.config["Data"]["normalized_data_maxs"]]) and \
                any([almost_equal(mis, data_min) for data_min in self.config["Data"]["normalized_data_mins"]]):
            return
        elif not (almost_equal(stds, 1.0) and almost_equal(avgs, 0.0)):
            msg = self.main_msgs['features_unnormalized']
            self.react(msg, )


class WeightsCheck(Debugger):

    def __init__(self, period):
        super().__init__("Weight_log")
        self.period = period
        self.actual_period = 0

    def run(self):
        self.actual_period += 1
        initial_weights, _ = get_model_weights_and_biases(self.model)
        layer_names = get_model_layer_names(self.model)
        for layer_name, weight_array in initial_weights.items():
            shape = weight_array.shape
            if len(shape) == 1 and shape[0] == 1:
                continue
            if almost_equal(np.var(weight_array), 0.0, rtol=1e-8):
                self.react(self.main_msgs['poor_init'].format(layer_name))
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
                try:
                    activation_layer = list(layer_names)[list(layer_names.keys()).index(layer_name) + 1]
                except Exception:
                    continue
                if isinstance(layer_names[activation_layer], torch.nn.ReLU) and not he_test:
                    abs_std_err = np.abs(np.std(weight_array) - np.sqrt(1.0 / fan_in))
                    self.react(self.main_msgs['need_he'].format(layer_name, abs_std_err))
                elif isinstance(layer_names[activation_layer], torch.nn.Tanh) and not glorot_test:
                    abs_std_err = np.abs(np.std(weight_array) - np.sqrt(2.0 / fan_in))
                    self.react(self.main_msgs['need_glorot'].format(layer_name, abs_std_err))
                elif isinstance(layer_names[activation_layer], torch.nn.Sigmoid) and not lecun_test:
                    abs_std_err = np.abs(np.std(weight_array) - np.sqrt(2.0 / (fan_in + fan_out)))
                    self.react(self.main_msgs['need_lecun'].format(layer_name, abs_std_err))
                elif not (lecun_test or he_test or glorot_test):
                    self.react(self.main_msgs['need_init_well'].format(layer_name))
