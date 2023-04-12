from abc import ABC
from typing import List
from hive.agents.agent import Agent
from hive.envs.base import BaseEnv

from hive.runners.utils import Metrics
from hive.utils import schedule
from hive.utils.experiment import Experiment
from hive.utils.loggers import ScheduledLogger
from hive.utils.registry import Registrable
from hive.utils.utils import seeder


class Runner(ABC, Registrable):
    """Base Runner class used to implement a training loop.

    Different types of training loops can be created by overriding the relevant
    functions.
    """

    def __init__(
        self,
        environment: BaseEnv,
        agents: List[Agent],
        logger: ScheduledLogger,
        experiment_manager: Experiment,
        train_steps: int,
        eval_environment: BaseEnv = None,
        test_frequency: int = -1,
        test_episodes: int = 1,
        max_steps_per_episode: int = 1e9,
    ):
        """
        Args:
            environment (BaseEnv): Environment used in the training loop.
            agents (list[Agent]): List of agents that interact with the environment.
            logger (ScheduledLogger): Logger object used to log metrics.
            experiment_manager (Experiment): Experiment object that saves the state of
                the training.
            train_steps (int): How many steps to train for. This is the number
                of times that agent.update is called. If this is -1, there is no
                limit for the number of training steps.
            test_frequency (int): After how many training steps to run testing episodes.
                If this is -1, testing is not run.
            test_episodes (int): How many episodes to run testing for.
        """
        self._train_environment = environment
        self._train_environment.seed(seeder.get_new_seed("environment"))
        self._eval_environment = eval_environment
        if self._eval_environment is not None:
            self._eval_environment.seed(seeder.get_new_seed("environment"))

        if isinstance(agents, list):
            self._agents = agents
        else:
            self._agents = [agents]
        self._logger = logger
        self._experiment_manager = experiment_manager
        if train_steps == -1:
            self._train_schedule = schedule.ConstantSchedule(True)
        else:
            self._train_schedule = schedule.SwitchSchedule(True, False, train_steps)
        if test_frequency == -1:
            self._test_schedule = schedule.ConstantSchedule(False)
        else:
            self._test_schedule = schedule.PeriodicSchedule(False, True, test_frequency)
        self._test_episodes = test_episodes
        self._max_steps_per_episode = max_steps_per_episode

        self._experiment_manager.register_experiment(
            logger=self._logger,
            agents=self._agents,
            environment=self._train_environment,
            eval_environment=self._eval_environment,
        )
        self._experiment_manager.experiment_state.update(
            {
                "train_schedule": self._train_schedule,
                "test_schedule": self._test_schedule,
            }
        )
        self._logger.register_timescale("train")
        self._logger.register_timescale("test")
        self._training = True
        self._save_experiment = False
        self._run_testing = False

    def register_config(self, config):
        self._experiment_manager.register_config(config)
        self._logger.log_config(config)

    def train_mode(self, training):
        """If training is true, sets all agents to training mode. If training is false,
        sets all agents to eval mode.

        Args:
            training (bool): Whether to be in training mode.
        """
        self._training = training
        for agent in self._agents:
            agent.train() if training else agent.eval()

    def create_episode_metrics(self):
        """Create the metrics used during the loop."""
        return Metrics(
            self._agents,
            [("reward", 0), ("episode_length", 0)],
            [("full_episode_length", 0)],
        )

    def update_step(self):
        """Update steps for various schedules. Run testing if appropriate."""
        if self._training:
            self._train_schedule.update()
            self._logger.update_step("train")
            if self._test_schedule.update():
                self.run_testing()
            self._save_experiment = (
                self._experiment_manager.update_step() or self._save_experiment
            )

    def run_episode(self, environment):
        """Run a single episode of the environment.

        Args:
            environment (BaseEnv): Environment in which the agent will take a step in.
        """
        return NotImplementedError

    def run_training(self):
        """Run the training loop. Note, to ensure that the test phase is run during
        the individual runners must call :py:meth:`~Runner.update_step` in their
        :py:meth:`~Runner.run_episode` methods.
        See :py:class:`~hive.runners.single_agent_loop.SingleAgentRunner` and
        :py:class:`~hive.runners.multi_agent_loop.MultiAgentRunner` for examples."""
        # Run an initial test episode
        self.run_testing()

        self.train_mode(True)
        while self._train_schedule.get_value():
            # Run training episode
            if not self._training:
                self.train_mode(True)
            episode_metrics = self.run_episode(self._train_environment)
            if self._logger.should_log("train"):
                episode_metrics = episode_metrics.get_flat_dict()
                self._logger.log_metrics(episode_metrics, "train")

            # Save experiment state
            if self._save_experiment:
                self._experiment_manager.save()
                self._save_experiment = False

        # Run a final test episode and save the experiment.
        self.run_testing()
        self._experiment_manager.save()

    def run_testing(self):
        """Run a testing phase."""
        if self._eval_environment is None:
            return
        self.train_mode(False)
        aggregated_episode_metrics = self.create_episode_metrics().get_flat_dict()
        for _ in range(self._test_episodes):
            episode_metrics = self.run_episode(self._eval_environment)
            for metric, value in episode_metrics.get_flat_dict().items():
                aggregated_episode_metrics[metric] += value / self._test_episodes
        self._logger.update_step("test")
        self._logger.log_metrics(aggregated_episode_metrics, "test")
        self._run_testing = False
        self.train_mode(True)

    def resume(self):
        """Resume a saved experiment."""
        if self._experiment_manager.is_resumable():
            self._experiment_manager.resume()
        self._train_schedule = self._experiment_manager.experiment_state[
            "train_schedule"
        ]
        self._test_schedule = self._experiment_manager.experiment_state["test_schedule"]

    @classmethod
    def type_name(cls):
        """
        Returns:
            "runner"
        """
        return "runner"
