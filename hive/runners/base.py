from abc import ABC, abstractmethod

from hive.utils import schedule
from hive.runners.utils import Metrics


class Runner(ABC):
    """Base Runner class used to implement a training loop.

    Different types of training loops can be created by overriding the relevant
    functions.
    """

    def __init__(
        self,
        environment,
        agents,
        logger,
        experiment_manager,
        train_steps=1000000,
        test_frequency=10000,
        test_steps=1,
        max_steps_per_episode=27000,
    ):
        """Initializes the Runner object.
        Args:
            environment: Environment used in the training loop.
            agents: List of agents that interact with the environment
            logger: Logger object used to log metrics.
            experiment_manager: ExperimentManager object that saves the state of the
                training.
            train_steps: How many steps to train for. If this is -1, there is no limit
                for the number of training steps.
            test_frequency: After how many training steps to run testing episodes.
                If this is -1, testing is not run.
            test_steps: How many steps to run testing for.
        """
        self._environment = environment
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
        self._test_steps = test_steps
        self._max_steps_per_episode = max_steps_per_episode

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

    def train_mode(self, training):
        """If training is true, sets all agents to training mode. If training is false,
        sets all agents to eval mode.
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

    def run_one_step(self, observation, turn, episode_metrics):
        """Run one step of the training loop.

        If it is the agent's first turn during the episode, do not run an update step.
        Otherwise, run an update step based on the previous action and accumulated
        reward since then.

        Args:
            observation: Current observation that the agent should create an action for.
            turn: Agent whose turn it is.
            episode_metrics: Metrics object keeping track of metrics for current episode.
        """
        if self._training:
            self._train_schedule.update()
            self._logger.update_step("train")
            self._run_testing = self._test_schedule.update() or self._run_testing
            self._save_experiment = (
                self._experiment_manager.update_step() or self._save_experiment
            )

    def run_end_step(self, episode_metrics, done):
        """Run the final step of an episode.

        After an episode ends, iterate through agents and update then with the final
        step in the episode.

        Args:
            episode_metrics: Metrics object keeping track of metrics for current episode.

        """
        return NotImplementedError

    def run_episode(self):
        """Run a single episode of the environment."""

        return NotImplementedError

    def run_training(self):
        """Run the training loop."""
        while self._train_schedule.get_value():
            # Run training episode
            episode_metrics, _ = self.run_episode()
            if self._logger.should_log("train"):
                episode_metrics = episode_metrics.get_flat_dict()
                self._logger.log_metrics(episode_metrics, "train")

            # Run test episodes
            if self._run_testing:
                test_metrics = self.run_testing()
                self._logger.update_step("test")
                self._logger.log_metrics(test_metrics, "test")
                self._run_testing = False

            # Save experiment state
            if self._save_experiment:
                self._experiment_manager.save()
                self._save_experiment = False

        # Run a final test episode and save the experiment.
        test_metrics = self.run_testing()
        self._logger.update_step("test")
        self._logger.log_metrics(test_metrics, "test")
        self._experiment_manager.save()

    def run_testing(self):
        self.train_mode(False)
        aggregated_episode_metrics = self.create_episode_metrics().get_flat_dict()
        test_steps = 0
        episodes = 0
        while test_steps <= self._test_steps:
            episode_metrics, steps = self.run_episode()
            test_steps += steps
            episodes += 1
            for metric, value in episode_metrics.get_flat_dict().items():
                aggregated_episode_metrics[metric] += value

        for metric in aggregated_episode_metrics:
            aggregated_episode_metrics[metric] /= episodes
        return aggregated_episode_metrics

    def resume(self):
        """Resume a saved experiment."""
        self._experiment_manager.resume()
        self._train_schedule = self._experiment_manager.experiment_state[
            "train_schedule"
        ]
        self._test_schedule = self._experiment_manager.experiment_state["test_schedule"]
