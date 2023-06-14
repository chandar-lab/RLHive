import logging
from abc import ABC
from typing import Optional, Sequence, Union, cast

from hive.agents.agent import Agent
from hive.envs.base import BaseEnv
from hive.utils.runner_utils import Metrics
from hive.types import Creates
from hive.utils import schedule
from hive.utils.config import Config, config_to_dict
from hive.utils.experiment import Experiment
from hive.utils.loggers import CompositeLogger, Logger, NullLogger, logger
from hive.utils.utils import Counter, Timer, seeder


class Runner(ABC):
    """Base Runner class used to implement a training loop.

    Different types of training loops can be created by overriding the relevant
    functions.
    """

    def __init__(
        self,
        environment: BaseEnv,
        agents: Sequence[Agent],
        loggers: Optional[Union[Creates[Logger], Sequence[Creates[Logger]]]],
        experiment_manager: Experiment,
        train_steps: int,
        eval_environment: Optional[BaseEnv] = None,
        test_frequency: int = -1,
        test_episodes: int = 1,
        max_steps_per_episode: int = 1_000_000_000,
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

        if isinstance(agents, Sequence):
            self._agents = agents
        else:
            self._agents = cast(Sequence[Agent], [agents])

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
        self._train_steps = Counter()
        self._phase_steps = Counter()
        self._test_steps = Counter()
        # Set up loggers
        if loggers is None:
            logger.set_logger(NullLogger())
        elif not isinstance(loggers, Sequence):
            logger.set_logger(loggers())
        else:
            logger.set_logger(CompositeLogger(loggers))
        logger.set_global_step(self._train_steps)

        self._experiment_manager.register_experiment(
            agents=self._agents,
            environment=self._train_environment,
            eval_environment=self._eval_environment,
        )
        self._experiment_manager.experiment_state.update(
            {
                "train_steps": self._train_steps,
                "test_steps": self._test_steps,
            }
        )
        self._training = True
        self._save_experiment = False
        self._run_testing = False
        self._full_timer = Timer()
        self._train_timer = Timer()
        self._eval_timer = Timer()
        self._full_timer.start()
        self._train_timer.start()

    def register_config(self, config: Config):
        """Register the config for the experiment.

        Args:
            config (Config): Config to register.
        """
        config_dict = config_to_dict(config)
        self._experiment_manager.register_config(config_dict)
        logger.log_config(config_dict)

    def train_mode(self, training):
        """If training is true, sets all agents to training mode. If training is false,
        sets all agents to eval mode.

        Args:
            training (bool): Whether to be in training mode.
        """
        if training and not self._training:
            self._train_timer.start()
            self._eval_timer.stop()
        elif not training and self._training:
            self._train_timer.stop()
            self._eval_timer.start()
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
            self._train_steps.increment()
            self._phase_steps.increment()
            if self._test_schedule(self._train_steps):
                self.run_testing()
                metrics = {
                    "current_fps": self._phase_steps / self._train_timer.get_time(),
                    "train_time": self._train_timer.get_time(),
                    "eval_time": self._eval_timer.get_time(),
                    "total_fps": self._train_steps / self._full_timer.get_time(),
                    "total_time": self._full_timer.get_time(),
                }
                logger.log_metrics(metrics, "progress")
                logging.info(
                    f"{self._train_steps.value} environment training steps completed\n"
                    + "\n".join([f"{k}: {v}" for k, v in metrics.items()])
                )
                self._phase_steps = Counter()
                self._train_timer.start()
                self.run_testing()
            if self._experiment_manager.should_save(self._train_steps):
                self._experiment_manager.save(self._train_steps)

    def run_episode(self, environment) -> Metrics:
        """Run a single episode of the environment.

        Args:
            environment (BaseEnv): Environment in which the agent will take a step in.
        """
        raise NotImplementedError

    def run_training(self):
        """Run the training loop. Note, to ensure that the test phase is run during
        the individual runners must call :py:meth:`~Runner.update_step` in their
        :py:meth:`~Runner.run_episode` methods.
        See :py:class:`~hive.runners.single_agent_loop.SingleAgentRunner` and
        :py:class:`~hive.runners.multi_agent_loop.MultiAgentRunner` for examples."""
        # Run an initial test episode
        self.run_testing()
        self.train_mode(True)
        while self._train_schedule(self._train_steps):
            # Run training episode
            if not self._training:
                self.train_mode(True)
            episode_metrics = self.run_episode(self._train_environment)
            episode_metrics = episode_metrics.get_flat_dict()
            logger.log_metrics(episode_metrics, "train")

        # Run a final test episode and save the experiment.
        self.run_testing()
        self._experiment_manager.save(self._train_steps)

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
        self._test_steps.increment()
        aggregated_episode_metrics["test_steps"] = self._test_steps.value
        logger.log_metrics(aggregated_episode_metrics, "test")
        self._run_testing = False
        self.train_mode(True)

    def resume(self):
        """Resume a saved experiment."""
        if self._experiment_manager.is_resumable():
            self._experiment_manager.resume()
        self._train_steps = cast(
            Counter, self._experiment_manager.experiment_state["train_steps"]
        )
        self._test_steps = cast(
            Counter, self._experiment_manager.experiment_state["test_steps"]
        )

        assert isinstance(self._train_steps, Counter)
        assert isinstance(self._test_steps, Counter)

    @classmethod
    def type_name(cls):
        """
        Returns:
            "runner"
        """
        return "runner"
