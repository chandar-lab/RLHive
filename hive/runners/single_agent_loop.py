import copy
from typing import List

from hive.agents.agent import Agent
from hive.envs.base import BaseEnv
from hive.runners import Runner

# from hive.runners.utils import TransitionInfo
from hive.utils import utils
from hive.utils.experiment import Experiment
from hive.utils.loggers import CompositeLogger, NullLogger, Logger
from gymnasium.vector.utils.numpy_utils import create_empty_array, concatenate
import numpy as np


class SingleAgentRunner(Runner):
    """Runner class used to implement a sinle-agent training loop."""

    def __init__(
        self,
        environment: BaseEnv,
        agent: Agent,
        loggers: List[Logger],
        experiment_manager: Experiment,
        train_steps: int,
        eval_environment: BaseEnv = None,
        test_frequency: int = -1,
        test_episodes: int = 1,
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
            train_steps (int): How many steps to train for. This is the number
                of times that agent.update is called. If this is -1, there is no
                limit for the number of training steps.
            eval_environment (BaseEnv): Environment used to evaluate the agent. If
                None, the ``environment`` parameter (which is a function) is
                used to create a second environment.
            test_frequency (int): After how many training steps to run testing
                episodes. If this is -1, testing is not run.
            test_episodes (int): How many episodes to run testing for duing each test
                phase.
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

        agent = agent(
            observation_space=env_spec.observation_space[0],
            action_space=env_spec.action_space[0],
        )

        # Set up experiment manager
        experiment_manager = experiment_manager()
        super().__init__(
            environment=environment,
            eval_environment=eval_environment,
            agents=[agent],
            loggers=loggers,
            experiment_manager=experiment_manager,
            train_steps=train_steps,
            test_frequency=test_frequency,
            test_episodes=test_episodes,
            max_steps_per_episode=max_steps_per_episode,
        )

    def run_one_step(
        self,
        environment: BaseEnv,
        observation,
        episode_metrics,
        agent_traj_state,
    ):
        """Run one step of the training loop.

        Args:
            observation: Current observation that the agent should create an action
                for.
            episode_metrics (Metrics): Keeps track of metrics for current episode.
        """
        agent = self._agents[0]
        action, agent_traj_state = agent.act(
            observation, agent_traj_state, self._train_steps
        )
        (
            next_observation,
            reward,
            terminated,
            truncated,
            _,
            other_info,
        ) = environment.step(action)

        info = {
            "observation": observation,
            "next_observation": next_observation,
            "reward": reward,
            "action": action,
            "terminated": terminated,
            "truncated": truncated,
            "info": other_info,
            "source": 0,
        }
        if self._training:
            agent_traj_state = agent.update(
                copy.deepcopy(info), agent_traj_state, self._train_steps
            )

        episode_metrics[agent.id]["reward"] += info["reward"]
        episode_metrics[agent.id]["episode_length"] += 1
        episode_metrics["full_episode_length"] += 1

        return terminated, truncated, next_observation, agent_traj_state

    def run_end_step(
        self,
        environment,
        observation,
        episode_metrics,
        agent_traj_state,
    ):
        """Run the final step of an episode.

        After an episode ends, set the truncated value to true.

        Args:
            environment (BaseEnv): Environment in which the agent will take a step in.
            observation: Current observation that the agent should create an action
                for.
            episode_metrics (Metrics): Keeps track of metrics for current
                episode.
            agent_traj_state: Trajectory state object that will be passed to the
                agent when act and update are called. The agent returns a new
                trajectory state object to replace the state passed in.

        """
        agent = self._agents[0]

        action, agent_traj_state = agent.act(observation, agent_traj_state)
        next_observation, reward, terminated, _, _, other_info = environment.step(
            action
        )
        truncated = not terminated

        info = {
            "observation": observation,
            "next_observation": next_observation,
            "reward": reward,
            "action": action,
            "terminated": terminated,
            "truncated": truncated,
            "info": other_info,
        }
        if self._training:
            agent_traj_state = agent.update(copy.deepcopy(info), agent_traj_state)

        episode_metrics[agent.id]["reward"] += info["reward"]
        episode_metrics[agent.id]["episode_length"] += 1
        episode_metrics["full_episode_length"] += 1

        return terminated, truncated, next_observation, agent_traj_state

    def run_episode(self, environment):
        """Run a single episode of the environment.

        Args:
            environment (BaseEnv): Environment in which the agent will take a step in.
        """
        episode_metrics = self.create_episode_metrics()
        terminated, truncated = False, False
        observation, _ = environment.reset()
        agent_traj_state = None
        steps = 0
        # Run the loop until the episode ends or times out
        while (
            not (terminated or truncated)
            and steps < self._max_steps_per_episode - 1
            and (not self._training or self._train_schedule(self._train_steps))
        ):
            terminated, truncated, observation, agent_traj_state = self.run_one_step(
                environment,
                observation,
                episode_metrics,
                agent_traj_state,
            )
            steps += 1
            self.update_step()

        if not (terminated or truncated):
            self.run_end_step(
                environment,
                observation,
                episode_metrics,
                agent_traj_state,
            )
            self.update_step()

        return episode_metrics
