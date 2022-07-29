from pathlib import Path
from hive.utils.registry import Registrable
from hive.debugger.utils import settings


# TODO: Debugger class needs to be divided in multiple classes :
#       1- Debugger class abstract class with the abstract methods
#       2- NullDebugger class (following the logger in RLHive logic) when the user doesn't want to use the Debugger
#       3- PreCheckDebugger class to run the pre-check properties
#       4- PostCheckDebugger class to run the post-check properties
#       5- OnTrainingCheckDebugger class to run the on-training-check properties
#       6- CompositeDebugger class to run the all-check properties
class Debugger(Registrable):
    def __init__(self, check_type, app_path=None):
        # TODO: merge this logger with RLHive's logger
        app_path = Path.cwd() if app_path == None else app_path
        log_fpath = settings.build_log_file_path(app_path, check_type)
        self.logger = settings.file_logger(log_fpath, check_type)
        # TODO: we need to follow the same config logic as RLHive !!!!
        config_fpath = settings.load_user_config_if_exists(app_path)
        self.config = settings.Config(config_fpath)
        self.pre_check = PreCheck(main_logger=self.logger, config=self.config.pre_check)

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
        self.pre_check.set_parameters(observations, model, labels, predictions, loss, opt, actions)

    def run_pre_checks(self, batch_size, implemented_ops):
        self.pre_check.run(batch_size, implemented_ops)


# TODO: This class should be named "PreCheckDebugger"
class PreCheck:

    def __init__(self, main_logger, config):
        # these parameters prevent running the pre_check multiple times
        self.pre_check_done = False
        self.main_logger = main_logger
        self.config = config
        self.main_msgs = settings.load_messages()

