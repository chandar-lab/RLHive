import abc
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

import wandb
from hive.types import Creates, PathLike
from hive.utils.registry import Creates, registry
from hive.utils.utils import Chomp, Counter, create_folder


class Logger(abc.ABC):
    """Abstract class for logging in hive."""

    def __init__(self) -> None:
        self._global_step = Counter()

    def set_global_step(self, global_step: Counter) -> None:
        self._global_step = global_step

    @abc.abstractmethod
    def log_config(self, config) -> None:
        """Log the config.

        Args:
            config (dict): Config parameters.
        """
        pass

    @abc.abstractmethod
    def log_scalar(self, name: str, value: Any, prefix: str) -> None:
        """Log a scalar variable.

        Args:
            name (str): Name of the metric to be logged.
            value (float): Value to be logged.
            prefix (str): Prefix to append to metric name.
        """
        pass

    @abc.abstractmethod
    def log_metrics(self, metrics: Mapping[str, Any], prefix: str) -> None:
        """Log a dictionary of values.

        Args:
            metrics (dict): Dictionary of metrics to be logged.
            prefix (str): Prefix to append to metric name.
        """
        pass

    def save(self, dir_name: PathLike) -> None:
        """Saves the current state of the log files.

        Args:
            dir_name (str): Name of the directory to save the log files.
        """
        pass

    def load(self, dir_name: PathLike) -> None:
        """Loads the log files from given directory.

        Args:
            dir_name (str): Name of the directory to load the log file from.
        """
        pass


class NullLogger(Logger):
    """A null logger that does not log anything.

    Used if you don't want to log anything, but still want to use parts of the
    framework that ask for a logger.
    """

    def log_config(self, config):
        pass

    def log_scalar(self, name, value, prefix):
        pass

    def log_metrics(self, metrics, prefix):
        pass


class WandbLogger(Logger):
    """A Wandb logger.

    This logger can be used to log to wandb. It assumes that wandb is configured
    locally on your system.

    Check the wandb documentation for more details on the parameters.
    """

    def __init__(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
        dir: Optional[str] = None,
        mode: Optional[str] = None,
        id: Optional[str] = None,
        resume: Optional[bool] = None,
        start_method: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
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
        super().__init__()
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
        wandb.config.update(dictify(config))

    def log_scalar(self, name, value, prefix):
        metrics = {f"{prefix}/{name}": value, "global_step": self._global_step.value}
        wandb.log(metrics, step=self._global_step.value)

    def log_metrics(self, metrics, prefix):
        metrics = {f"{prefix}/{name}": value for (name, value) in metrics.items()}
        metrics["global_step"] = self._global_step.value
        wandb.log(metrics, step=self._global_step.value)


def dictify(config):
    if isinstance(config, list):
        return {idx: dictify(param) for idx, param in enumerate(config)}
    elif isinstance(config, dict):
        return {k: dictify(v) for k, v in config.items()}
    else:
        return config


class ChompLogger(Logger):
    """This logger uses the Chomp data structure to store all logged values which are
    then directly saved to disk.
    """

    def __init__(self):
        super().__init__()
        self._log_data = Chomp()

    def log_config(self, config):
        self._log_data["config"] = config

    def log_scalar(self, name, value, prefix):
        metric_name = f"{prefix}/{name}"
        if self._global_step.value not in self._log_data:
            self._log_data[self._global_step.value] = {}
        if isinstance(value, torch.Tensor):
            self._log_data[self._global_step.value][metric_name] = value.item()
        else:
            self._log_data[self._global_step.value][metric_name] = value

    def log_metrics(self, metrics, prefix):
        if self._global_step.value not in self._log_data:
            self._log_data[self._global_step.value] = {}
        for name in metrics:
            metric_name = f"{prefix}/{name}"
            if isinstance(metrics[name], torch.Tensor):
                self._log_data[self._global_step.value][metric_name] = metrics[
                    name
                ].item()
            else:
                self._log_data[self._global_step.value][metric_name] = metrics[name]

    def save(self, dir_name):
        super().save(dir_name)
        self._log_data.save(Path(dir_name) / "log_data.p")

    def load(self, dir_name):
        super().load(dir_name)
        self._log_data.load(Path(dir_name) / "log_data.p")


class CompositeLogger(Logger):
    """This Logger aggregates multiple loggers together.

    This logger is for convenience and allows for logging using multiple loggers without
    having to keep track of several loggers. When the `global_step` is updated, this
    logger updates the `global_step` for each one of its component loggers. When
    logging, logs to each of its component loggers.
    """

    def __init__(self, logger_list: Sequence[Creates[Logger]]):
        super().__init__()
        self._logger_list = [logger_fn() for logger_fn in logger_list]

    def set_global_step(self, global_step: Counter):
        super().set_global_step(global_step)
        for logger in self._logger_list:
            logger.set_global_step(global_step)

    def log_config(self, config):
        for logger in self._logger_list:
            logger.log_config(config)

    def log_scalar(self, name, value, prefix):
        for logger in self._logger_list:
            logger.log_scalar(name, value, prefix)

    def log_metrics(self, metrics, prefix):
        for logger in self._logger_list:
            logger.log_metrics(metrics, prefix=prefix)

    def save(self, dir_name):
        path = Path(dir_name)
        for idx, logger in enumerate(self._logger_list):
            save_dir = path / f"logger_{idx}"
            create_folder(save_dir)
            logger.save(save_dir)

    def load(self, dir_name):
        path = Path(dir_name)
        for idx, logger in enumerate(self._logger_list):
            logger.load(path / f"logger_{idx}")


registry.register_classes(
    {
        "NullLogger": NullLogger,
        "WandbLogger": WandbLogger,
        "ChompLogger": ChompLogger,
        "CompositeLogger": CompositeLogger,
    },
)


class GlobalLoggerWrapper(Logger):
    def __init__(self):
        self._logger: Logger = NullLogger()

    def set_logger(self, logger: Logger):
        self._logger = logger

    def set_global_step(self, global_step: Counter):
        self._global_step = global_step
        self._logger.set_global_step(global_step)

    def log_config(self, config):
        self._logger.log_config(config)

    def log_scalar(self, name, value, prefix):
        self._logger.log_scalar(name, value, prefix)

    def log_metrics(self, metrics, prefix):
        self._logger.log_metrics(metrics, prefix)

    def save(self, dir_name):
        self._logger.save(dir_name)

    def load(self, dir_name):
        self._logger.load(dir_name)


logger: GlobalLoggerWrapper = GlobalLoggerWrapper()
