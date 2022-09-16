from pathlib import Path
import numpy as np

from hive.utils.registry import Registrable
from hive.debugger.utils import settings


def almost_equal(value1, value2, rtol=1e-2):
    rerr = np.abs(value1 - value2)
    if isinstance(value1, np.ndarray):
        return (rerr <= rtol).all()
    else:
        return rerr <= rtol


class Debugger:
    def __init__(self, main_logger='/', config='utils/config/settings.yaml', app_path=None):
        # TODO: merge this logger with RLHive's logger
        app_path = Path.cwd() if app_path == None else app_path
        log_fpath = settings.build_log_file_path(app_path, 'log.txt')
        self.logger = settings.file_logger(log_fpath, 'log.txt')
        # TODO: we need to follow the same config logic as RLHive !!!!
        config_fpath = settings.load_user_config_if_exists(app_path)
        self.config = settings.Config(config_fpath)
        self.main_msgs = settings.load_messages()
        self.main_logger = main_logger
        self.config = config
        self.main_msgs = settings.load_messages()

    def set_debugger(self, config, full_config):
        debugger_config = config["debugger"]
        debugger_fn, full_config["debugger"] = debugger_lib.get_debugger(debugger_config, "debugger")
        self.debuggers = debugger_fn()

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

    def run_pre_checks(self, batch_size, implemented_ops):
        self.pre_check.run(batch_size, implemented_ops)

    def react(self, message):
        if self.config.fail_on:
            self.main_logger.error(message)
            raise Exception(message)
        else:
            self.main_logger.warning(message)


class ObservationsCheck(Debugger, Registrable):

    def __init__(self):
        Debugger.__init__()

    def run(self):
        mas = np.max(self.observations)
        mis = np.min(self.observations)
        avgs = np.mean(self.observations)
        stds = np.std(self.observations)

        # for idx in range(len(mas)):
        if stds == 0.0:
            msg = self.main_msgs['features_constant']
            self.react(msg)
        elif any([almost_equal(mas, data_max) for data_max in self.config.data.normalized_data_maxs]) and \
                any([almost_equal(mis, data_min) for data_min in self.config.data.normalized_data_mins]):
            return
        elif not (almost_equal(stds, 1.0) and almost_equal(avgs, 0.0)):
            msg = self.main_msgs['features_unnormalized']
            self.react(msg)
