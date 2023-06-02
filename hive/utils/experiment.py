"""Implementation of a simple experiment class."""
import logging
from pathlib import Path
import shutil
from typing import Union, Sequence

import yaml

from hive.agents.agent import Agent
from hive.envs.base import BaseEnv
from hive.utils.loggers import logger
from hive.utils.registry import Registrable, registry
from hive.utils.schedule import Schedule
from hive.utils.utils import Chomp, Counter, create_folder


class Experiment(Registrable):
    """Implementation of a simple experiment class."""

    def __init__(
        self,
        name: str,
        save_dir: str,
        saving_schedule: Schedule,
        num_checkpoints_to_save: int = 1,
    ):
        """Initializes an experiment object.

        The experiment state is an exposed property of objects of this class. It can
        be used to keep track of objects that need to be saved to keep track of the
        experiment, but don't fit in one of the standard categories. One example of
        this is the various schedules used in the Runner class.

        Args:
            name (str): Name of the experiment.
            dir_name (str): Absolute path to the directory to save/load the experiment.
            saving_schedule (Schedule): Schedule that determines when the
                experiment should be saved.
        """

        self._name = name
        self._save_dir = Path(save_dir) / name
        self._saving_schedule = saving_schedule()
        self._num_checkpoints_to_save = num_checkpoints_to_save
        self._checkpoints_saved = []
        create_folder(self._save_dir)

        self._config = None
        self.experiment_state = Chomp()
        self._experiment_components = {}

    def register_experiment(
        self,
        agents: Union[Agent, Sequence[Agent]],
        environment: BaseEnv,
        **kwargs,
    ):
        """Registers all the components of an experiment.

        Args:
            logger (Logger): a logger object.
            agents (Agent | list[Agent]): either an agent object or a list of agents.
            environment (BaseEnv): an environment object.
        """

        self._experiment_components.update(
            {"logger": logger, "agents": agents, "environment": environment, **kwargs}
        )

    def register_config(self, config):
        """Registers the experiment config.

        Args:
            config (Chomp): a config dictionary.
        """
        self._config = config

    def should_save(self, step: Counter):
        """Returns whether you should save the experiment at the current step."""
        return self._saving_schedule(step)

    def save(self, tag="current"):
        """Saves the experiment.
        Args:
            tag (str): Tag to prefix the folder.
        """
        save_dir = self._save_dir / str(tag)
        create_folder(save_dir)

        logging.info("Saving the experiment at {}".format(save_dir))

        flag_file = save_dir / "flag.p"

        if flag_file.is_file():
            flag_file.unlink()

        if self._config is not None:
            file_name = save_dir / "config.yml"
            with file_name.open("w") as f:
                yaml.safe_dump(dict(self._config), f)

        save_component(self._experiment_components, save_dir)

        file_name = save_dir / "experiment_state.p"
        self.experiment_state.save(file_name)

        flag_file.touch()
        self._checkpoints_saved.append(save_dir)
        if len(self._checkpoints_saved) > self._num_checkpoints_to_save:
            shutil.rmtree(self._checkpoints_saved.pop(0))

    def is_resumable(self, tag="current"):
        """Returns true if the experiment is resumable.

        Args:
            tag (str): Tag for the saved experiment.
        """

        flag_file = self._save_dir / str(tag) / "flag.p"
        return flag_file.is_file()

    def resume(self, tag="current"):
        """Resumes the experiment from a checkpoint.

        Args:
            tag (str): Tag for the saved experiment.
        """

        if self.is_resumable(tag):
            save_dir = Path(self._save_dir) / str(tag)
            logging.info("Loading the experiment from {}".format(save_dir))

            if self._config is not None:
                file_name = save_dir / "config.yml"
                with file_name.open() as f:
                    self._config = Chomp(yaml.safe_load(f))

            load_component(self._experiment_components, save_dir)
            file_name = save_dir / "experiment_state.p"
            self.experiment_state.load(file_name)
            self._saving_schedule = self.experiment_state["saving_schedule"]

    @classmethod
    def type_name(cls):
        """
        Returns:
            "experiment_manager"
        """
        return "experiment_manager"


def save_component(component, prefix):
    if component is None:
        return
    elif isinstance(component, list):
        for idx, sub_component in enumerate(component):
            save_component(sub_component, prefix / str(idx))
    elif isinstance(component, dict):
        for name, sub_component in component.items():
            save_component(sub_component, prefix / name)
    elif hasattr(component, "save") and callable(getattr(component, "save")):
        create_folder(prefix)
        try:
            save_fn = getattr(component, "save")
            save_fn(prefix)
            logging.info(f"Saved {component} to {prefix}")
        except NotImplementedError:
            logging.info(f"{component} save not implemented")


def load_component(component, prefix: Path):
    if component is None:
        return
    elif isinstance(component, list):
        for idx, sub_component in enumerate(component):
            load_component(sub_component, prefix / str(idx))
    elif isinstance(component, dict):
        for name, sub_component in component.items():
            load_component(sub_component, prefix / name)
    elif hasattr(component, "load") and callable(getattr(component, "load")):
        if prefix.is_dir():
            try:
                load_fn = getattr(component, "load")
                load_fn(prefix)
                logging.info(f"Loaded {component} from {prefix}")
            except:
                logging.info(f"{component} not loaded")


registry.register("Experiment", Experiment, Experiment)
