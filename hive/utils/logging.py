import os
import abc
import copy
import wandb
from hive.utils.schedule import ConstantSchedule, Schedule, get_schedule
from hive.utils.utils import Chomp, create_folder, create_class_constructor


class Logger(abc.ABC):
    """Abstract class for logging in hive."""

    def __init__(self, timescales):
        """Constructor for base Logger class. Every Logger must call this constructor
        in its own constructor
        
        Args:
            timescales (str|List): The different timescales at which logger needs to log.
                If only logging at one timescale, it is acceptable to only pass a string. 
        """
        if isinstance(timescales, str):
            self._timescales = [timescales]
        elif isinstance(timescales, list):
            self._timescales = timescales
        else:
            raise ValueError("Need string or list of strings for timescales")

    def register_timescale(self, timescale):
        """Register a new timescale with the logger.
        
        Args:
            timescale (str): timescale to register.
        """
        self._timescales.append(timescale)

    @abc.abstractmethod
    def log_scalar(self, name, value, timescale):
        """Log a scalar variable.
        Args:
            name: str, name of the metric to be logged.
            value: float, value to be logged.
            step: int, step value.
            timescale (str): timescale at which to log.
        """
        pass

    @abc.abstractmethod
    def log_metrics(self, metrics, timescale):
        """Log a scalar variable.
        Args:
            metrics: dict, dictionary of metrics to be logged.
            step: int, step value.
            timescale (str): timescale at which to log.
        """
        pass

    @abc.abstractmethod
    def save(self, dir_name):
        """Saves the current state of the log files.
        Args:
            dir_name: name of the directory to save the log files.
        """
        pass

    @abc.abstractmethod
    def load(self, dir_name):
        """Loads the log files from given directory.
        Args:
            dir_name: name of the directory to load the log file from.
        """
        pass


class ScheduledLogger(Logger):
    """Abstract class that manages a schedule for logging.
    
    The update_step method should be called for each step in the loop to update
    the logger's schedule. The should_log method can be used to check whether
    the logger should log anything. 

    This schedule is not strictly enforced! It is still possible to log something
    even if should_log returns false. These functions are just for the purpose
    of convenience.

    """

    def __init__(self, timescales, logger_schedules=None):
        """Constructor for abstract class ScheduledLogger. Should be called by
        each subclass in the constructor.
        
        Any timescales not assigned schedule from logger_schedules will be assigned
        a ConstantSchedule(True).
        Args:
            timescales (str|List): The different timescales at which logger needs to log.
                If only logging at one timescale, it is acceptable to only pass a string.
            logger_schedules (Schedule|list|dict): Schedules used to keep track of when 
                to log. If a single schedule, it is copied for each timescale. If a list
                of schedules, the schedules are matched up in order with the list of
                timescales provided. If a dictionary, the keys should be the timescale
                and the values should be the schedule.
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
            timescale (str): timescale to register.
            schedule: Schedule to use for this timescale. 
        """
        super().register_timescale(timescale)
        if schedule is None:
            schedule = ConstantSchedule(True)
        self._logger_schedules[timescale] = schedule
        self._steps[timescale] = 0

    def update_step(self, timescale):
        """Update the step and schedule for a given timescale."""
        self._steps[timescale] += 1
        self._logger_schedules[timescale].update()
        return self.should_log(timescale)

    def should_log(self, timescale):
        """Check if you should log for a given timescale."""
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

    def __init__(self, **kwargs):
        super().__init__("null", ConstantSchedule(False))

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
    """

    def __init__(
        self,
        project_name,
        run_name,
        timescales="wandb",
        logger_schedules=None,
        offline=False,
        **kwargs,
    ):
        """Constructor for the WandbLogger.
        
        Args:
            project_name (str): Name of the project. Wandb's dash groups all runs with 
                the same project name together. 
            run_name (str): Name of the run. Used to identify the run on the wandb dash.
            logger_schedule (Schedule): Schedule used to define when logging should occur.
            logger_name (str): Used to differentiate between different loggers/timescales
                in the same run.
            offline (bool): Whether to log offline. 
        """
        super().__init__(timescales, logger_schedules)
        wandb.init(project=project_name, name=run_name)

    def log_scalar(self, name, value, timescale):
        metrics = {f"{timescale}_{name}": value}
        metrics.update(
            {
                f"_{timescale}_step": self._steps[timescale]
                for timescale in self._timescales
            }
        )
        wandb.log(metrics)

    def log_metrics(self, metrics, timescale):
        metrics = {f"{timescale}_{name}": value for (name, value) in metrics.items()}
        metrics.update(
            {
                f"_{timescale}_step": self._steps[timescale]
                for timescale in self._timescales
            }
        )
        wandb.log(metrics)


class ChompLogger(ScheduledLogger):
    """This logger uses the Chomp data structure to store all logged values which are then
    directly saved to disk.
    """

    def __init__(self, timescales, logger_schedules=None):
        super().__init__(timescales, logger_schedules)
        self._log_data = Chomp()

    def log_scalar(self, name, value, timescale):
        metric_name = f"{timescale}_{name}"
        if self._log_data[metric_name] is None:
            self._log_data[metric_name] = [[], []]
        self._log_data[metric_name][0].append(value)
        self._log_data[metric_name][1].append(
            [self._steps[timescale] for timescale in self._timescales]
        )

    def log_metrics(self, metrics, timescale):
        for name in metrics:
            metric_name = f"{timescale}_{name}"
            if self._log_data[metric_name] is None:
                self._log_data[metric_name] = [[], []]
            self._log_data[metric_name][0].append(metrics[name])
            self._log_data[metric_name][1].append(
                [self._steps[timescale] for timescale in self._timescales]
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

    def __init__(self, logger_list):
        super().__init__([])
        self._logger_list = logger_list
        for idx, logger in enumerate(self._logger_list):
            if isinstance(logger, dict):
                self._logger_list[idx] = get_logger(logger)

    def register_timescale(self, timescale, schedule=None):
        for logger in self._logger_list:
            if isinstance(logger, ScheduledLogger):
                logger.register_timescale(timescale, schedule)
            else:
                logger.register_timescale(timescale)

    def log_scalar(self, name, value, timescale):
        for logger in self._logger_list:
            if not isinstance(logger, ScheduledLogger) or logger.should_log(timescale):
                logger.log_scalar(name, value, timescale=timescale)

    def log_metrics(self, metrics, timescale):
        for logger in self._logger_list:
            if not isinstance(logger, ScheduledLogger) or logger.should_log(timescale):
                logger.log_metrics(metrics, timescale=timescale)

    def update_step(self, timescale):
        for logger in self._logger_list:
            if isinstance(logger, ScheduledLogger):
                logger.update_step(timescale)
        return self.should_log(timescale)

    def should_log(self, timescale):
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


get_logger = create_class_constructor(
    Logger,
    {
        "NullLogger": NullLogger,
        "WandbLogger": WandbLogger,
        "ChompLogger": ChompLogger,
        "CompositeLogger": CompositeLogger,
    },
)
