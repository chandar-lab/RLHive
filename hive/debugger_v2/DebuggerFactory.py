from pathlib import Path
from hive import debugger_v2 as debugger_lib
from hive.debugger_v2.utils.model_params_getters import get_model_weights_and_biases, get_model_layer_names
from hive.debugger_v2.utils import settings
import inspect


def period(check_period, iter_num):
    return ((check_period != 0) and (iter_num % check_period)) \
           or ((check_period == 0) and (iter_num == -1))


class DebuggerFactory:
    def __init__(self, app_path=None):
        # TODO: Clean this please
        app_path = Path.cwd() if app_path == None else app_path
        log_fpath = settings.build_log_file_path(app_path, "logger")
        self.logger = settings.file_logger(log_fpath, "logger")
        # TODO: we need to follow the same config logic as RLHive !!!!
        config_fpath = settings.load_user_config_if_exists(app_path)
        self.config = settings.Config(config_fpath).full_conf
        self.debuggers = dict()

        self.params = {}

    def set_debugger(self, config):
        for debugger_config in config:
            debugger_fn, _ = debugger_lib.get_debugger(debugger_config, debugger_config["name"])
            debugger = debugger_fn(self.config[debugger_config["name"]]["Period"])
            debugger.config = self.config[debugger_config["name"]]
            self.debuggers[debugger_config["name"]] = debugger

    def set_parameters(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def react(self, message, fail_on=False):
        if fail_on:
            self.logger.error(message)
            raise Exception(message)
        else:
            self.logger.warning(message)

    def run(self):
        for debugger in self.debuggers.values():
            if period(check_period=debugger.check_period, iter_num=debugger.iter_num):
                args = inspect.getfullargspec(debugger.run).args[1:]
                kwargs = {arg: self.params[arg] for arg in args}
                msg = debugger.run(**kwargs)
                self.react(msg)







                if debugger.check_type == "Observation":
                    msg = debugger.run(self.params["observations"])
                    self.react(msg)

                if debugger.check_type == "Weight":
                    initial_weights, _ = get_model_weights_and_biases(self.params["model"])
                    layer_names = get_model_layer_names(self.params["model"])
                    msg = debugger.run(initial_weights, layer_names)
                    self.react(msg)

                if debugger.check_type == "Missing Reset State":
                    msg = debugger.run(self.params["observations"])
                    self.react(msg)