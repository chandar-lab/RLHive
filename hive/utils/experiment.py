"""Implementation of a simple experiment class."""
import logging
import os

import yaml
from hive.utils.registry import Registrable, registry
from hive.utils.schedule import Schedule

from hive.utils.utils import Chomp, create_folder


class Experiment(Registrable):
    """Implementation of a simple experiment class."""

    def __init__(self, name: str, save_dir: str, saving_schedule: Schedule):
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
        self._save_dir = os.path.join(save_dir, name)
        self._saving_schedule = saving_schedule()
        self._step = 0
        create_folder(self._save_dir)

        self._config = None
        self.experiment_state = Chomp()
        self.experiment_state["saving_schedule"] = self._saving_schedule
        self._experiment_components = {}

    def register_experiment(self, **kwargs):
        """Registers all the components of an experiment.

        Args:
            logger (Logger): a logger object.
            agents (Agent | list[Agent]): either an agent object or a list of agents.
            environment (BaseEnv): an environment object.
        """

        self._experiment_components.update(kwargs)

    def register_config(self, config):
        """Registers the experiment config.

        Args:
            config (Chomp): a config dictionary.
        """
        self._config = config

    def update_step(self):
        """Updates the step of the saving schedule for the experiment."""
        self._step += 1
        return self._saving_schedule.update()

    def should_save(self):
        """Returns whether you should save the experiment at the current step."""
        return self._saving_schedule.get_value()

    def save(self, tag="current"):
        """Saves the experiment.
        Args:
            tag (str): Tag to prefix the folder.
        """

        save_dir = os.path.join(self._save_dir, tag)
        create_folder(save_dir)

        logging.info("Saving the experiment at {}".format(save_dir))

        flag_file = os.path.join(save_dir, "flag.p")
        if os.path.isfile(flag_file):
            os.remove(flag_file)

        if self._config is not None:
            file_name = os.path.join(save_dir, "config.yml")
            with open(file_name, "w") as f:
                yaml.safe_dump(dict(self._config), f)

        save_component(self._experiment_components, save_dir)

        file_name = os.path.join(save_dir, "experiment_state.p")
        self.experiment_state.save(file_name)

        file = open(flag_file, "w")
        file.close()

    def is_resumable(self, tag="current"):
        """Returns true if the experiment is resumable.

        Args:
            tag (str): Tag for the saved experiment.
        """

        flag_file = os.path.join(self._save_dir, tag, "flag.p")
        if os.path.isfile(flag_file):
            return True
        else:
            return False

    def resume(self, tag="current"):
        """Resumes the experiment from a checkpoint.

        Args:
            tag (str): Tag for the saved experiment.
        """

        if self.is_resumable(tag):
            save_dir = os.path.join(self._save_dir, tag)
            logging.info("Loading the experiment from {}".format(save_dir))

            if self._config is not None:
                file_name = os.path.join(save_dir, "config.yml")
                with open(file_name) as f:
                    self._config = Chomp(yaml.safe_load(f))

            load_component(self._experiment_components, save_dir)
            file_name = os.path.join(save_dir, "experiment_state.p")
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
            save_component(sub_component, os.path.join(prefix, str(idx)))
    elif isinstance(component, dict):
        for name, sub_component in component.items():
            save_component(sub_component, os.path.join(prefix, name))
    elif hasattr(component, "save") and callable(getattr(component, "save")):
        folder_name = os.path.join(prefix)
        create_folder(folder_name)
        try:
            save_fn = getattr(component, "save")
            save_fn(folder_name)
        except NotImplementedError:
            logging.info(f"{prefix} save not implemented")


def load_component(component, prefix):
    if component is None:
        return
    elif isinstance(component, list):
        for idx, sub_component in enumerate(component):
            load_component(sub_component, os.path.join(prefix, idx))
    elif isinstance(component, dict):
        for name, sub_component in component.items():
            load_component(sub_component, os.path.join(prefix, name))
    elif hasattr(component, "load") and callable(getattr(component, "load")):
        folder_name = os.path.join(prefix)
        create_folder(folder_name)
        try:
            load_fn = getattr(component, "load")
            load_fn(folder_name)
        except:
            logging.info(f"{prefix} not loaded")


registry.register("Experiment", Experiment, Experiment)
