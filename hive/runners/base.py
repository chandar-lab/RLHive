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
        train_steps,
        train_episodes,
        test_frequency,
        test_num_episodes,
    ):
        """Initializes the Runner object.
        Args:
            environment: Environment used in the training loop.
            agents: List of agents that interact with the environment
            logger: Logger object used to log metrics.
            experiment_manager: ExperimentManager object that saves the state of the
                training.
            train_steps: How many steps to train for. If this is -1, there is no limit
                for the number of training steps. If both this and train_episodes are
                -1, training loop will not terminate.
            train_episodes: How many episodes to train for. If this is -1, there is no
                limit for the number of training episodes. If both this and train_steps
                are -1, training loop will not terminate.
            test_frequency: After how many training episodes to run testing episodes.
                If this is -1, testing is not run.
            test_num_episodes: How many testing episodes to run during each testing
                period.
        """
        self._environment = environment
        if isinstance(agents, list):
            self._agents = agents
        else:
            self._agents = [agents]
        self._logger = logger
        self._experiment_manager = experiment_manager
        if train_steps == -1:
            self._train_step_schedule = schedule.ConstantSchedule(True)
        else:
            self._train_step_schedule = schedule.SwitchSchedule(
                True, False, train_steps
            )
        if train_episodes == -1:
            self._train_episode_schedule = schedule.ConstantSchedule(True)
        else:
            self._train_episode_schedule = schedule.SwitchSchedule(
                True, False, train_episodes
            )
        if test_frequency == -1:
            self._test_schedule = schedule.ConstantSchedule(False)
        else:
            self._test_schedule = schedule.PeriodicSchedule(False, True, test_frequency)
        self.test_num_episodes = test_num_episodes
        self._train_step_schedule.update()
        self._test_schedule.update()
        self._experiment_manager.experiment_state.add_from_dict(
            {
                "train_step_schedule": self._train_step_schedule,
                "train_episode_schedule": self._train_episode_schedule,
                "test_schedule": self._test_schedule,
            }
        )

    def train_mode(self, training):
        """If training is true, sets all agents to training mode. If training is false,
        sets all agents to eval mode.
        """
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
        return NotImplementedError

    def run_end_step(self, episode_metrics):
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

        while (
            self._train_episode_schedule.update()
            and self._train_step_schedule.get_value()
        ):
            # Run training episode
            self.train_mode(True)
            episode_metrics = self.run_episode()
            if self._logger.update_step("train_episodes"):
                self._logger.log_metrics(
                    episode_metrics.get_flat_dict(), "train_episodes"
                )

            # Run test episodes
            while self._test_schedule.update():
                self.train_mode(False)
                mean_episode_metrics = dict.fromkeys(episode_metrics.get_flat_dict(), 0)
                for _ in range(self.test_num_episodes):
                    episode_metrics = self.run_episode()
                    for metric, value in episode_metrics.get_flat_dict().items():
                        mean_episode_metrics[metric] += value / self.test_num_episodes
                if self._logger.update_step("test_episodes"):
                    self._logger.log_metrics(mean_episode_metrics, "test_episodes")

            # Save experiment state
            if self._experiment_manager.update_step():
                self._experiment_manager.save()

        # Run a final test episode and save the experiment.
        self.train_mode(False)
        mean_episode_metrics = dict.fromkeys(episode_metrics.get_flat_dict(), 0)
        for _ in range(self.test_num_episodes):
            episode_metrics = self.run_episode()
            for metric, value in episode_metrics.get_flat_dict().items():
                mean_episode_metrics[metric] += value / self.test_num_episodes

        self._logger.update_step("test_episodes")
        self._logger.log_metrics(mean_episode_metrics, "test_episodes")
        self._experiment_manager.save()

    def resume(self):
        """Resume a saved experiment."""
        self._experiment_manager.resume()
        self._train_step_schedule = self._experiment_manager.experiment_state[
            "train_step_schedule"
        ]
        self._train_episode_schedule = self._experiment_manager.experiment_state[
            "train_episode_schedule"
        ]
        self._test_schedule = self._experiment_manager.experiment_state["test_schedule"]
