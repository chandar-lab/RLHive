import copy
import time
from typing import List

import numpy as np
from gymnasium.vector.utils.numpy_utils import concatenate, create_empty_array

from hive.agents.agent import Agent
from hive.envs.base import BaseEnv
from hive.envs.parallel_env import AsyncEnv
from hive.runners.base import Runner
from hive.runners.utils import Metrics, TransitionInfo
from hive.utils import utils
from hive.utils.experiment import Experiment
from hive.utils.loggers import logger


class TransitionInfo:
    def __init__(self, observation_space, action_space, num_envs=1):
        self.observation = create_empty_array(observation_space, n=num_envs)
        self.action = create_empty_array(action_space, n=num_envs)
        self.reward = np.zeros(num_envs)
        self.info = np.empty(num_envs, dtype=object)
        self.started = np.zeros(num_envs, dtype=bool)

    def get_info(self, indices=None):
        if indices is None:
            indices = 0
        return {
            "observation": self.observation[indices],
            "action": self.action[indices],
            "reward": self.reward[indices],
            "info": self.info[indices],
            "source": indices,
        }

    def update(self, indices=None, **kwargs):
        if indices is None:
            indices = 0
        for key, value in kwargs.items():
            getattr(self, key)[indices] = value

    def is_started(self, indices=None):
        if indices is None:
            indices = 0
        return self.started[indices]

    def start(self, indices=None):
        if indices is None:
            indices = 0
        self.started[indices] = True

    def finish(self, indices=None):
        if indices is None:
            indices = [0]
        self.reward[indices] = 0
        self.started[indices] = False

        # self.next_observations = create_empty_array(observation_space, n=num_envs)
        # self.terminations = np.zeros(num_envs)
        # self.truncations = np.zeros(num_envs)


class ParallelSingleAgentRunner(Runner):
    """Runner class used to implement a sinle-agent training loop."""

    def __init__(
        self,
        environment: BaseEnv,
        num_envs: int,
        agent: Agent,
        loggers: List[ScheduledLogger],
        experiment_manager: Experiment,
        train_steps: int,
        batch_size: int = -1,
        eval_environment: BaseEnv = None,
        test_frequency: int = -1,
        test_episodes: int = 1,
        max_steps_per_episode: int = 1e9,
        seed: int = None,
    ):
        """Initializes the Runner object.

        Args:
            environment (BaseEnv): Environment used in the training loop.
            agent (Agent): Agent that will interact with the environment
            logger (ScheduledLogger): Logger object used to log metrics.
            experiment_manager (Experiment): Experiment object that saves the state of
                the training.
            train_steps (int): How many steps to train for. If this is -1, there is no
                limit for the number of training steps.
            test_frequency (int): After how many training steps to run testing
                episodes. If this is -1, testing is not run.
            test_episodes (int): How many episodes to run testing for duing each test
                phase.
            stack_size (int): The number of frames in an observation sent to an agent.
            max_steps_per_episode (int): The maximum number of steps to run an episode
                for.
        """
        if seed is not None:
            utils.seeder.set_global_seed(seed)
        if eval_environment is None:
            eval_environment = environment
        if batch_size == -1:
            batch_size = num_envs
        environment = environment(num_envs=num_envs, batch_size=batch_size)
        eval_environment = (
            eval_environment(num_envs=test_episodes, batch_size=test_episodes)
            if test_frequency != -1
            else None
        )
        self._env_spec = environment.env_spec
        # Set up loggers
        if loggers is None:
            logger = NullLogger()
        else:
            logger = CompositeLogger(loggers)

        agent = agent(
            observation_space=self._env_spec.observation_space[0],
            action_space=self._env_spec.action_space[0],
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
        self._transition_info = TransitionInfo(
            self._env_spec.observation_space[0],
            self._env_spec.action_space[0],
            num_envs,
        )
        self._metrics = utils.Chomp(
            {
                "return": np.zeros(num_envs),
                "episode_length": np.zeros(num_envs, dtype=int),
            }
        )
        self._eval_metrics = utils.Chomp(
            {
                "return": np.zeros(self._test_episodes),
                "episode_length": np.zeros(self._test_episodes, dtype=int),
            }
        )
        self._num_envs = num_envs
        # Slow down generating process if it's too far ahead of updating process
        # self.

    def run_training(self):
        self.train_mode(True)
        self._train_environment.reset()
        all_agent_traj_states = np.empty(self._num_envs, dtype=object)
        while self._train_schedule(global_step):
            finished = self.run_step(
                all_agent_traj_states=all_agent_traj_states,
                metrics=self._metrics,
                transition_info=self._transition_info,
                environment=self._train_environment,
            )
            self.finish_step(
                transition_info=self._transition_info,
                metrics=self._metrics,
                finished=finished,
                timescale="train",
            )
            self.update_step()
        self.run_testing()
        self._experiment_manager.save()
        self._train_environment.close()
        self._eval_environment.close()

    def run_testing(self):
        if self._eval_environment is None:
            return
        self.train_mode(False)
        num_environments = self._test_episodes
        self._eval_environment.reset()
        all_agent_traj_states = np.empty(num_environments, dtype=object)
        eval_metrics = utils.Chomp(
            {
                "return": np.zeros(self._test_episodes),
                "episode_length": np.zeros(self._test_episodes, dtype=int),
            }
        )
        step_metrics = utils.Chomp(
            {
                "return": np.zeros(self._test_episodes),
                "episode_length": np.zeros(self._test_episodes, dtype=int),
            }
        )
        finished = np.array([], dtype=int)
        transition_info = TransitionInfo(
            self._env_spec.observation_space[0],
            self._env_spec.action_space[0],
            num_environments,
        )

        while len(finished) < num_environments:
            finished_step = self.run_step(
                all_agent_traj_states=all_agent_traj_states,
                metrics=step_metrics,
                transition_info=transition_info,
                environment=self._eval_environment,
            )
            newly_finished = np.setdiff1d(finished_step, finished)
            eval_metrics["return"][newly_finished] = step_metrics["return"][
                newly_finished
            ]
            eval_metrics["episode_length"][newly_finished] = step_metrics[
                "episode_length"
            ][newly_finished]
            step_metrics["return"][finished_step] = 0
            step_metrics["episode_length"][finished_step] = 0
            transition_info.finish(finished_step)
            finished = np.append(finished, newly_finished)
        logger.update_step("test")
        logger.log_metrics(
            {
                "return": np.mean(eval_metrics["return"]),
                "episode_length": np.mean(eval_metrics["episode_length"]),
            },
            "test",
        )
        self.train_mode(True)

    def run_step(self, all_agent_traj_states, metrics, transition_info, environment):
        agent = self._agents[0]
        env_ids, (
            observations,
            rewards,
            terminated,
            truncated,
            _,
            infos,
        ) = environment.get_next_observations()
        started = transition_info.is_started(env_ids)
        num_started = np.sum(started)

        if num_started > 0:
            started_env_ids = env_ids[started]
            agent_traj_states = all_agent_traj_states[started_env_ids]
            if self._training:
                update_info = transition_info.get_info(started_env_ids)
                update_info.update(
                    {
                        "next_observation": observations[started],
                        "reward": rewards[started],
                        "terminated": terminated[started],
                        "truncated": truncated[started],
                        # "next_info": infos[started_env_ids],
                    }
                )
                all_agent_traj_states[started_env_ids] = agent.update(
                    copy.deepcopy(update_info), agent_traj_states
                )
                self.update_step()
            metrics["return"][started_env_ids] += rewards[started]
            metrics["episode_length"][started_env_ids] += 1
        finished = np.logical_or(terminated, truncated)

        agent_traj_states = all_agent_traj_states[env_ids]
        actions, agent_traj_states = agent.act(observations, agent_traj_states)
        transition_info.update(
            env_ids,
            observation=observations,
            action=actions,  # info=infos
        )
        environment.send(actions, env_ids)

        transition_info.start(env_ids[np.logical_not(started)])
        return env_ids[finished]

    def finish_step(
        self,
        transition_info,
        metrics,
        finished,
        timescale="train",
    ):
        for env_id in finished:
            if logger.update_step(timescale):
                logger.log_metrics(
                    {
                        "return": self._metrics["return"][env_id],
                        "episode_length": self._metrics["episode_length"][env_id],
                    },
                    timescale,
                )
        metrics["return"][finished] = 0
        metrics["episode_length"][finished] = 0
        transition_info.finish(finished)

    def update_step(self):
        if self._training:
            self._train_steps.increment()
            if self._experiment_manager.update_step():
                self._experiment_manager.save()
            if self._test_schedule(self._train_steps):
                self.run_testing()
