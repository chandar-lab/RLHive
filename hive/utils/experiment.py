"""Implementation of a simple experiment class."""
import logging
import os
from shutil import rmtree, copyfile

from hive.utils.utils import create_folder, Chomp


class Experiment(object):
    """Implementation of a simple experiment class."""

    def __init__(self, name, dir_name, schedule):
        """Initializes an experiment object.

        Args:
            name: str, name of the experiment.
            dir_name: str, absolute path to the directory to save/load the experiment.
        """

        self._name = name
        self._dir_name = os.path.join(dir_name, name)
        self._schedule = schedule
        self._step = 0
        create_folder(self._dir_name)

        self._config = None
        self._logger = None
        self._agents = None
        self._environment = None
        self.experiment_state = Chomp()
        self.experiment_state["saving_schedule"] = self._schedule

    def register_experiment(
        self, config=None, logger=None, agents=None, environment=None,
    ):
        """Registers all the components of an experiment.

        Args:
            config: a config dictionary.
            logger: a logger object.
            experiment_state: a experiment_state dictionary.
            agents: either an agent object or a list of agents.
            environment: an environment object.
        """

        self._config = config
        self._logger = logger
        if agents is not None and not isinstance(agents, list):
            agents = [agents]
        self._agents = agents
        self._environment = environment

    def update_step(self):
        self._step += 1
        return self._schedule.update()

    def should_save(self):
        return self._schedule.get_value()

    def save(self, tag="current"):
        """Saves the experiment.
        Args:
            tag: str, tag to prefix the folder.
        """

        save_dir = os.path.join(self._dir_name, tag)
        create_folder(save_dir)

        logging.info("Saving the experiment at {}".format(save_dir))

        flag_file = os.path.join(save_dir, "flag.p")
        if os.path.isfile(flag_file):
            os.remove(flag_file)

        if self._config is not None:
            file_name = os.path.join(save_dir, "config.p")
            self._config.save(file_name)

        if self._logger is not None:
            folder_name = os.path.join(save_dir, "logger")
            create_folder(folder_name)
            self._logger.save(folder_name)

        if self._agents is not None:
            for idx, agent in enumerate(self._agents):
                agent_dir = os.path.join(save_dir, f"agent_{idx}")
                create_folder(agent_dir)
                agent.save(agent_dir)

        if self._environment is not None:
            file_name = os.path.join(save_dir, "environment.p")
            self._environment.save(file_name)

        file_name = os.path.join(save_dir, "experiment_state.p")
        self.experiment_state.save(file_name)

        file = open(flag_file, "w")
        file.close()

    def is_resumable(self, tag="current"):
        """ Returns true if the experiment is resumable.

        Args:
            tag: str, tag for the saved experiment.
        """

        flag_file = os.path.join(self._dir_name, tag, "flag.p")
        if os.path.isfile(flag_file):
            return True
        else:
            return False

    def resume(self, tag="current"):
        """Resumes the experiment from a checkpoint.

        Args:
            tag: str, tag for the saved experiment.
        """

        if not self.is_resumable(tag):
            logging.warning("This experiment is not resumable!")
            logging.warning("Force restarting the experiment!")
            self.force_restart(tag)

        else:
            save_dir = os.path.join(self._dir_name, tag)
            logging.info("Loading the experiment from {}".format(save_dir))

            if self._config is not None:
                file_name = os.path.join(save_dir, "config.p")
                self._config.load(file_name)

            if self._logger is not None:
                folder_name = os.path.join(save_dir, "logger")
                self._logger.load(folder_name)

            if self._agents is not None:
                for idx, agent in enumerate(self._agents):
                    agent_dir = os.path.join(save_dir, f"agent_{idx}")
                    agent.load(agent_dir)

            if self._environment is not None:
                file_name = os.path.join(save_dir, "environment.p")
                self._environment.load(file_name)

            file_name = os.path.join(save_dir, "experiment_state.p")
            self.experiment_state.load(file_name)
            self._schedule = self.experiment_state["saving_schedule"]

    def force_restart(self):
        """Force restarting an experiment from beginning."""

        logging.info("Force restarting the experiment...")

        save_dir = os.path.join(self._dir_name)
        create_folder(save_dir)
        rmtree(save_dir)

        if self._logger is not None:
            self._logger.force_restart()
