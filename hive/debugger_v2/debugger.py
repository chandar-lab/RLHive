# from pathlib import Path
# import torch.nn
# import numpy as np
# from hive import debugger_v2 as debugger_lib
# from hive.debugger_v2.utils.model_params_getters import get_model_weights_and_biases, get_model_layer_names
# from hive.utils.registry import Registrable
# from hive.debugger_v2.utils import settings
# from hive.debugger_v2.utils import metrics
#
#
# class Debugger(Registrable):
#     def __init__(self, logger='log.txt', config='utils/config/settings.yaml', app_path=None):
#         # TODO: merge this logger with RLHive's logger
#         app_path = Path.cwd() if app_path == None else app_path
#         log_fpath = settings.build_log_file_path(app_path, logger)
#         self.logger = settings.file_logger(log_fpath, logger)
#         # TODO: we need to follow the same config logic as RLHive !!!!
#         config_fpath = settings.load_user_config_if_exists(app_path)
#         self.config = settings.Config(config_fpath).full_conf
#         self.main_msgs = settings.load_messages()
#         self.main_msgs = settings.load_messages()
#         self.debuggers = dict()
#
#         self.observations = None
#         self.model = None
#         self.labels = None
#         self.predictions = None
#         self.loss = None
#         self.opt = None
#         self.actions = None
#
#     def set_debugger(self, config):
#         for debugger_config in config:
#             debugger_fn, _ = debugger_lib.get_debugger(debugger_config, debugger_config["name"])
#             debugger = debugger_fn(self.config[debugger_config["name"]]["Period"])
#             debugger.config = self.config[debugger_config["name"]]
#             self.debuggers[debugger_config["name"]] = debugger
#
#     @classmethod
#     def type_name(cls):
#         return "debugger"
#
#     def set_parameters(self, observations, model, labels, predictions, loss, opt, actions):
#         self.observations = observations
#         self.model = model
#         self.labels = labels
#         self.predictions = predictions
#         self.loss = loss
#         self.opt = opt
#         self.actions = actions
#
#     def react(self, message, fail_on=False):
#         if fail_on:
#             self.main_logger.error(message)
#             raise Exception(message)
#         else:
#             self.logger.warning(message)
#
#     def run(self):
#         for debugger in self.debuggers.values():
#             if ((debugger.period != 0) and (debugger.actual_period % debugger.period)) \
#                     or ((debugger.period == 0) and (debugger.actual_period == -1)):
#                 debugger.set_parameters(self.observations, self.model, self.labels, self.predictions,
#                                         self.loss, self.opt, self.actions)
#                 debugger.run()
