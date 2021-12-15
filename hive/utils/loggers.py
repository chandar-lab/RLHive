import abc
import copy
import os
from typing import List

import torch
import wandb

from hive.utils.registry import Registrable, registry
from hive.utils.schedule import ConstantSchedule, Schedule, get_schedule
from hive.utils.utils import Chomp, create_folder


class Logger(abc.ABC, Registrable):
    """Abstract class for logging in hive."""

    def __init__(self, timescales=None):
        """Constructor for base Logger class. Every Logger must call this constructor
        in its own constructor

        Args:
            timescales (str | list(str)): The different timescales at which logger
                needs to log. If only logging at one timescale, it is acceptable to
                only pass a string.
        """
        if timescales is None:
            self._timescales = []
        elif isinstance(timescales, str):
            self._timescales = [timescales]
        elif isinstance(timescales, list):
            self._timescales = timescales
        else:
            raise ValueError("Need string or list of strings for timescales")

    def register_timescale(self, timescale):
        """Register a new timescale with the logger.

        Args:
            timescale (str): Timescale to register.
        """
        self._timescales.append(timescale)

    @abc.abstractmethod
    def log_config(self, config):
        """Log the config.

        Args:
            config (dict): Config parameters.
        """
        pass

    @abc.abstractmethod
    def log_scalar(self, name, value, prefix):
        """Log a scalar variable.

        Args:
            name (str): Name of the metric to be logged.
            value (float): Value to be logged.
            prefix (str): Prefix to append to metric name.
        """
        pass

    @abc.abstractmethod
    def log_metrics(self, metrics, prefix):
        """Log a dictionary of values.

        Args:
            metrics (dict): Dictionary of metrics to be logged.
            prefix (str): Prefix to append to metric name.
        """
        pass

    @abc.abstractmethod
    def save(self, dir_name):
        """Saves the current state of the log files.

        Args:
            dir_name (str): Name of the directory to save the log files.
        """
        pass

    @abc.abstractmethod
    def load(self, dir_name):
        """Loads the log files from given directory.

        Args:
            dir_name (str): Name of the directory to load the log file from.
        """
        pass

    @classmethod
    def type_name(cls):
        return "logger"


class ScheduledLogger(Logger):
    """Abstract class that manages a schedule for logging.

    The update_step method should be called for each step in the loop to update
    the logger's schedule. The should_log method can be used to check whether
    the logger should log anything.

    This schedule is not strictly enforced! It is still possible to log something
    even if should_log returns false. These functions are just for the purpose
    of convenience.
    """

    def __init__(self, timescales=None, logger_schedules=None):
        """
        Any timescales not assigned schedule from logger_schedules will be assigned
        a ConstantSchedule(True).

        Args:
            timescales (str|list[str]): The different timescales at which logger needs
                to log. If only logging at one timescale, it is acceptable to only pass
                a string.
            logger_schedules (Schedule|list|dict): Schedules used to keep track of when
                to log. If a single schedule, it is copied for each timescale. If a
                list of schedules, the schedules are matched up in order with the list
                of timescales provided. If a dictionary, the keys should be the
                timescale and the values should be the schedule.
        """
        super().__init__(timescales)
        if logger_schedules is None:
            logger_schedules = ConstantSchedule(True)
        if isinstance(logger_schedules, dict):
            self._logger_schedules = logger_schedules
        elif isinstance(logger_schedules, list):
            self._logger_schedules = {
                self._timescales[idx]: logger_schedules[idx]
                for idx in range(min(len(logger_schedules), len(self._timescales)))
            }
        elif isinstance(logger_schedules, Schedule):
            self._logger_schedules = {
                timescale: copy.deepcopy(logger_schedules)
                for timescale in self._timescales
            }
        else:
            raise ValueError(
                "logger_schedule must be a dict, list of Schedules, or Schedule object"
            )
        for timescale, schedule in self._logger_schedules.items():
            if isinstance(schedule, dict):
                self._logger_schedules[timescale] = get_schedule(
                    schedule["name"], schedule["kwargs"]
                )

        for timescale in self._timescales:
            if timescale not in self._logger_schedules:
                self._logger_schedules[timescale] = ConstantSchedule(True)
        self._steps = {timescale: 0 for timescale in self._timescales}

    def register_timescale(self, timescale, schedule=None):
        """Register a new timescale.

        Args:
            timescale (str): Timescale to register.
            schedule (Schedule): Schedule to use for this timescale.
        """
        super().register_timescale(timescale)
        if schedule is None:
            schedule = ConstantSchedule(True)
        self._logger_schedules[timescale] = schedule
        self._steps[timescale] = 0

    def update_step(self, timescale):
        """Update the step and schedule for a given timescale.

        Args:
            timescale (str): A registered timescale.
        """
        self._steps[timescale] += 1
        self._logger_schedules[timescale].update()
        return self.should_log(timescale)

    def should_log(self, timescale):
        """Check if you should log for a given timescale.

        Args:
            timescale (str): A registered timescale.
        """
        return self._logger_schedules[timescale].get_value()

    def save(self, dir_name):
        logger_state = Chomp()
        logger_state.timescales = self._timescales
        logger_state.schedules = self._logger_schedules
        logger_state.steps = self._steps
        logger_state.save(os.path.join(dir_name, "logger_state.p"))

    def load(self, dir_name):
        logger_state = Chomp()
        logger_state.load(os.path.join(dir_name, "logger_state.p"))
        self._timescales = logger_state.timescales
        self._logger_schedules = logger_state.schedules
        self._steps = logger_state.steps


class NullLogger(ScheduledLogger):
    """A null logger that does not log anything.

    Used if you don't want to log anything, but still want to use parts of the
    framework that ask for a logger.
    """

    def __init__(self, timescales=None, logger_schedules=None):
        super().__init__(timescales, logger_schedules)

    def log_config(self, config):
        pass

    def log_scalar(self, name, value, timescale):
        pass

    def log_metrics(self, metrics, timescale):
        pass

    def save(self, dir_name):
        pass

    def load(self, dir_name):
        pass


class WandbLogger(ScheduledLogger):
    """A Wandb logger.

    This logger can be used to log to wandb. It assumes that wandb is configured
    locally on your system. Multiple timescales/loggers can be implemented by
    instantiating multiple loggers with different logger_names. These should still
    have the same project and run names.

    Check the wandb documentation for more details on the parameters.
    """

    def __init__(
        self,
        timescales=None,
        logger_schedules=None,
        project=None,
        name=None,
        dir=None,
        mode=None,
        id=None,
        resume=None,
        start_method=None,
        **kwargs,
    ):
        """
        Args:
            timescales (str|list[str]): The different timescales at which logger needs
                to log. If only logging at one timescale, it is acceptable to only pass
                a string.
            logger_schedules (Schedule|list|dict): Schedules used to keep track of when
                to log. If a single schedule, it is copied for each timescale. If a
                list of schedules, the schedules are matched up in order with the list
                of timescales provided. If a dictionary, the keys should be the
                timescale and the values should be the schedule.
            project (str): Name of the project. Wandb's dash groups all runs with
                the same project name together.
            name (str): Name of the run. Used to identify the run on the wandb
                dash.
            dir (str): Local directory where wandb saves logs.
            mode (str): The mode of logging. Can be "online", "offline" or "disabled".
                In offline mode, writes all data to disk for later syncing to a server,
                while in disabled mode, it makes all calls to wandb api's noop's, while
                maintaining core functionality.
            id (str, optional): A unique ID for this run, used for resuming.
                It must be unique in the project, and if you delete a run you can't
                reuse the ID.
            resume (bool, str, optional): Sets the resuming behavior.
                Options are the same as mentioned in Wandb's doc.
            start_method (str): The start method to use for wandb's process. See
                https://docs.wandb.ai/guides/track/launch#init-start-error.
            **kwargs: You can pass any other arguments to wandb's init method as
                keyword arguments. Note, these arguments can't be overriden from the
                command line.
        """
        super().__init__(timescales, logger_schedules)

        settings = None
        if start_method is not None:
            settings = wandb.Settings(start_method=start_method)

        wandb.init(
            project=project,
            name=name,
            dir=dir,
            mode=mode,
            id=id,
            resume=resume,
            settings=settings,
            **kwargs,
        )

    def log_config(self, config):
        # Convert list parameters to nested dictionary
        for k, v in config.items():
            if isinstance(v, list):
                config[k] = {}
                for idx, param in enumerate(v):
                    config[k][idx] = param

        wandb.config.update(config)

    def log_scalar(self, name, value, prefix):
        metrics = {f"{prefix}/{name}": value}
        metrics.update(
            {
                f"{timescale}_step": self._steps[timescale]
                for timescale in self._timescales
            }
        )
        wandb.log(metrics)

    def log_metrics(self, metrics, prefix):
        metrics = {f"{prefix}/{name}": value for (name, value) in metrics.items()}
        metrics.update(
            {
                f"{timescale}_step": self._steps[timescale]
                for timescale in self._timescales
            }
        )
        wandb.log(metrics)


class ChompLogger(ScheduledLogger):
    """This logger uses the Chomp data structure to store all logged values which are
    then directly saved to disk.
    """

    def __init__(self, timescales=None, logger_schedules=None):
        super().__init__(timescales, logger_schedules)
        self._log_data = Chomp()

    def log_config(self, config):
        self._log_data["config"] = config

    def log_scalar(self, name, value, prefix):
        metric_name = f"{prefix}/{name}"
        if metric_name not in self._log_data:
            self._log_data[metric_name] = [[], []]
        if isinstance(value, torch.Tensor):
            self._log_data[metric_name][0].append(value.item())
        else:
            self._log_data[metric_name][0].append(value)
        self._log_data[metric_name][1].append(
            {timescale: self._steps[timescale] for timescale in self._timescales}
        )

    def log_metrics(self, metrics, prefix):
        for name in metrics:
            metric_name = f"{prefix}/{name}"
            if metric_name not in self._log_data:
                self._log_data[metric_name] = [[], []]
            if isinstance(metrics[name], torch.Tensor):
                self._log_data[metric_name][0].append(metrics[name].item())
            else:
                self._log_data[metric_name][0].append(metrics[name])
            self._log_data[metric_name][1].append(
                {timescale: self._steps[timescale] for timescale in self._timescales}
            )

    def save(self, dir_name):
        super().save(dir_name)
        self._log_data.save(os.path.join(dir_name, "log_data.p"))

    def load(self, dir_name):
        super().load(dir_name)
        self._log_data.load(os.path.join(dir_name, "log_data.p"))


class CompositeLogger(Logger):
    """This Logger aggregates multiple loggers together.

    This logger is for convenience and allows for logging using multiple loggers without
    having to keep track of several loggers. When timescales are updated, this logger
    updates the timescale for each one of its component loggers. When logging, logs to
    each of its component loggers as long as the logger is not a ScheduledLogger that
    should not be logging for the timescale.
    """

    def __init__(self, logger_list: List[Logger]):
        super().__init__([])
        self._logger_list = logger_list

    def register_timescale(self, timescale, schedule=None):
        for logger in self._logger_list:
            if isinstance(logger, ScheduledLogger):
                logger.register_timescale(timescale, schedule)
            else:
                logger.register_timescale(timescale)

    def log_config(self, config):
        for logger in self._logger_list:
            logger.log_config(config)

    def log_scalar(self, name, value, prefix):
        for logger in self._logger_list:
            logger.log_scalar(name, value, prefix)

    def log_metrics(self, metrics, prefix):
        for logger in self._logger_list:
            logger.log_metrics(metrics, prefix=prefix)

    def update_step(self, timescale):
        """Update the step and schedule for a given timescale for every
        ScheduledLogger.

        Args:
            timescale (str): A registered timescale.
        """

        for logger in self._logger_list:
            if isinstance(logger, ScheduledLogger):
                logger.update_step(timescale)
        return self.should_log(timescale)

    def should_log(self, timescale):
        """Check if you should log for a given timescale. If any logger in the list
        is scheduled to log, returns True.

        Args:
            timescale (str): A registered timescale.
        """
        for logger in self._logger_list:
            if not isinstance(logger, ScheduledLogger) or logger.should_log(timescale):
                return True
        return False

    def save(self, dir_name):
        for idx, logger in enumerate(self._logger_list):
            save_dir = os.path.join(dir_name, f"logger_{idx}")
            create_folder(save_dir)
            logger.save(save_dir)

    def load(self, dir_name):
        for idx, logger in enumerate(self._logger_list):
            load_dir = os.path.join(dir_name, f"logger_{idx}")
            logger.load(load_dir)


registry.register_all(
    Logger,
    {
        "NullLogger": NullLogger,
        "WandbLogger": WandbLogger,
        "ChompLogger": ChompLogger,
        "CompositeLogger": CompositeLogger,
    },
)

get_logger = getattr(registry, f"get_{Logger.type_name()}")
