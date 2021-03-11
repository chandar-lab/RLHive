import abc
import wandb
from hive.utils.schedule import ConstantSchedule


class Logger(abc.ABC):
    """Abstract class for logging in hive."""

    @abc.abstractmethod
    def log_scalar(self, name, value):
        """Log a scalar variable.
        Args:
            name: str, name of the metric to be logged.
            value: float, value to be logged.
            step: int, step value.
        """
        pass

    @abc.abstractmethod
    def log_metrics(self, metrics):
        """Log a scalar variable.
        Args:
            metrics: dict, dictionary of metrics to be logged.
            step: int, step value.
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

    def __init__(self, logger_schedule):
        """Constructor for abstract class ScheduledLogger. Should be called by
        each subclass in the constructor.
        
        Args:
            logger_schedule: A schedule used to define when logging should occur.
        """
        self._logger_schedule = logger_schedule

    def update_step(self):
        self._logger_schedule.update()
        return self.should_log()

    def should_log(self):
        return self._logger_schedule.get_value()


class NullLogger(ScheduledLogger):
    """A null logger that does not log anything. 
    
    Used if you don't want to log anything, but still want to use parts of the
    framework that ask for a logger.
    """

    def __init__(self):
        super().__init__(ConstantSchedule(False))

    def log_scalar(self, name, value):
        pass

    def log_metrics(self, metrics):
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
        self, project_name, run_name, logger_schedule=None, logger_name="wandb",
    ):
        """Constructor for the WandbLogger.
        
        Args:
            project_name (str): Name of the project. Wandb's dash groups all runs with 
                the same project name together. 
            run_name (str): Name of the run. Used to identify the run on the wandb dash.
            logger_schedule (Schedule): Schedule used to define when logging should occur.
            logger_name (str): Used to differentiate between different loggers/timescales
                in the same run.
        """
        if logger_schedule is None:
            logger_schedule = ConstantSchedule(True)
        super().__init__(logger_schedule)
        self._logger_name = logger_name
        self._step = 0
        self._step_name = f"{logger_name}_step"
        wandb.init(project=project_name, name=run_name)

    def update_step(self):
        self._step += 1
        return super().update_step()

    def log_scalar(self, name, value):
        wandb.log({f"{self._logger_name}_{name}": value, self._step_name: self._step})

    def log_metrics(self, metrics):
        metrics = {
            f"{self._logger_name}_{name}": value for (name, value) in metrics.items()
        }
        metrics[self._step_name] = self._step
        wandb.log(metrics)

    def save(self, dir_name):
        pass

    def load(self, dir_name):
        pass
