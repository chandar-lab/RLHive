import copy
from typing import List

from hive.agents.agent import Agent
from hive.envs.base import BaseEnv
from hive.runners import Runner
from hive.runners.utils import TransitionInfo
from hive.utils import utils
from hive.utils.experiment import Experiment
from hive.utils.loggers import CompositeLogger, NullLogger, ScheduledLogger


class SingleAgentRunner(Runner):
    """Runner class used to implement a sinle-agent training loop."""

    def __init__(
        self,
        environment: BaseEnv,
        agent: Agent,
        loggers: List[ScheduledLogger],
        experiment_manager: Experiment,
        train_steps: int,
        eval_environment: BaseEnv = None,
        test_frequency: int = -1,
        test_episodes: int = 1,
        stack_size: int = 1,
        max_steps_per_episode: int = 1e9,
        seed: int = None,
    ):
        """Initializes the SingleAgentRunner.

        Args:
            environment (BaseEnv): Environment used in the training loop.
            agent (Agent): Agent that will interact with the environment
            loggers (List[ScheduledLogger]): List of loggers used to log metrics.
            experiment_manager (Experiment): Experiment object that saves the state of
                the training.
            train_steps (int): How many steps to train for. If this is -1, there is no
                limit for the number of training steps.
            eval_environment (BaseEnv): Environment used to evaluate the agent. If
                None, the ``environment`` parameter (which is a function) is
                used to create a second environment.
            test_frequency (int): After how many training steps to run testing
                episodes. If this is -1, testing is not run.
            test_episodes (int): How many episodes to run testing for duing each test
                phase.
            stack_size (int): The number of frames in an observation sent to an agent.
            max_steps_per_episode (int): The maximum number of steps to run an episode
                for.
            seed (int): Seed used to set the global seed for libraries used by
                Hive and seed the :py:class:`~hive.utils.utils.Seeder`.
        """
        if seed is not None:
            utils.seeder.set_global_seed(seed)
        if eval_environment is None:
            eval_environment = environment
        environment = environment()
        eval_environment = eval_environment() if test_frequency != -1 else None
        env_spec = environment.env_spec
        # Set up loggers
        if loggers is None:
            logger = NullLogger()
        else:
            logger = CompositeLogger(loggers)

        agent = agent(
            observation_space=env_spec.observation_space[0],
            action_space=env_spec.action_space[0],
            stack_size=stack_size,
            logger=logger,
        )

        # Set up experiment manager
        experiment_manager = experiment_manager()
        super().__init__(
            environment=environment,
            eval_environment=eval_environment,
            agents=[agent],
            logger=logger,
            experiment_manager=experiment_manager,
            train_steps=train_steps,
            test_frequency=test_frequency,
            test_episodes=test_episodes,
            max_steps_per_episode=max_steps_per_episode,
        )
        self._stack_size = stack_size

    def run_one_step(self, environment, observation, episode_metrics, transition_info):
        """Run one step of the training loop.

        Args:
            observation: Current observation that the agent should create an action
                for.
            episode_metrics (Metrics): Keeps track of metrics for current episode.
        """
        super().run_one_step(environment, observation, 0, episode_metrics)
        agent = self._agents[0]
        stacked_observation = transition_info.get_stacked_state(agent, observation)
        action = agent.act(stacked_observation)
        next_observation, reward, done, _, other_info = environment.step(action)

        info = {
            "observation": observation,
            "reward": reward,
            "action": action,
            "done": done,
            "info": other_info,
        }
        if self._training:
            agent.update(copy.deepcopy(info))

        transition_info.record_info(agent, info)
        episode_metrics[agent.id]["reward"] += info["reward"]
        episode_metrics[agent.id]["episode_length"] += 1
        episode_metrics["full_episode_length"] += 1

        return done, next_observation

    def run_episode(self, environment):
        """Run a single episode of the environment."""
        episode_metrics = self.create_episode_metrics()
        done = False
        observation, _ = environment.reset()
        transition_info = TransitionInfo(self._agents, self._stack_size)
        transition_info.start_agent(self._agents[0])
        steps = 0
        # Run the loop until the episode ends or times out
        while (
            not done
            and steps < self._max_steps_per_episode
            and (not self._training or self._train_schedule.get_value())
        ):
            done, observation = self.run_one_step(
                environment, observation, episode_metrics, transition_info
            )
            steps += 1
            if self._run_testing and self._training:
                # Run test episodes
                self.run_testing()

        return episode_metrics
