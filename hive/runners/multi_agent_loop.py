import copy
from typing import List

from hive.agents.agent import Agent
from hive.envs.base import BaseEnv
from hive.runners.base import Runner
from hive.runners.utils import TransitionInfo
from hive.utils import utils
from hive.utils.experiment import Experiment
from hive.utils.loggers import CompositeLogger, NullLogger, ScheduledLogger


class MultiAgentRunner(Runner):
    """Runner class used to implement a multiagent training loop."""

    def __init__(
        self,
        environment: BaseEnv,
        agents: List[Agent],
        loggers: List[ScheduledLogger],
        experiment_manager: Experiment,
        train_steps: int,
        num_agents: int,
        eval_environment: BaseEnv = None,
        test_frequency: int = -1,
        test_episodes: int = 1,
        stack_size: int = 1,
        self_play: bool = False,
        max_steps_per_episode: int = 1e9,
        seed: int = None,
    ):
        """Initializes the MultiAgentRunner object.

        Args:
            environment (BaseEnv): Environment used in the training loop.
            agent (Agent): Agent that will interact with the environment
            loggers (List[ScheduledLogger]): List of loggers used to log metrics.
            experiment_manager (Experiment): Experiment object that saves the state of
                the training.
            train_steps (int): How many steps to train for. If this is -1, there is no
                limit for the number of training steps.
            num_agents (int): Number of agents running in this multiagent experiment.
            eval_environment (BaseEnv): Environment used to evaluate the agent. If
                None, the ``environment`` parameter (which is a function) is
                used to create a second environment.
            test_frequency (int): After how many training steps to run testing
                episodes. If this is -1, testing is not run.
            test_episodes (int): How many episodes to run testing for duing each test
                phase.
            stack_size (int): The number of frames in an observation sent to an agent.
            self_play (bool): Whether this multiagent experiment is run in
                self-play mode. In this mode, only the first agent in the list
                of agents provided in the config is created. This agent performs
                actions for each player in the multiagent environment.
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

        agent_list = []
        num_agents = num_agents if self_play else len(agents)
        for idx in range(num_agents):
            if not self_play or idx == 0:
                agent_fn = agents[idx]
                agent = agent_fn(
                    observation_space=env_spec.observation_space[idx],
                    action_space=env_spec.action_space[idx],
                    stack_size=stack_size,
                    logger=logger,
                )
                agent_list.append(agent)
            else:
                agent_list.append(copy.copy(agent_list[0]))
                agent_list[-1]._id = f"{agent_list[0]._id}_{idx}"

        # Set up experiment manager
        experiment_manager = experiment_manager()

        super().__init__(
            environment=environment,
            eval_environment=eval_environment,
            agents=agent_list,
            logger=logger,
            experiment_manager=experiment_manager,
            train_steps=train_steps,
            test_frequency=test_frequency,
            test_episodes=test_episodes,
            max_steps_per_episode=max_steps_per_episode,
        )
        self._stack_size = stack_size
        self._self_play = self_play

    def run_one_step(
        self,
        environment,
        observation,
        turn,
        episode_metrics,
        transition_info,
        agent_states,
    ):
        """Run one step of the training loop.

        If it is the agent's first turn during the episode, do not run an update step.
        Otherwise, run an update step based on the previous action and accumulated
        reward since then.

        Args:
            observation: Current observation that the agent should create an action
                for.
            turn (int): Agent whose turn it is.
            episode_metrics (Metrics): Keeps track of metrics for current episode.
        """
        self.update_runner_state()
        agent = self._agents[turn]
        agent_state = agent_states[turn]
        if transition_info.is_started(agent):
            info = transition_info.get_info(agent)
            if self._training:
                agent_state = agent.update(copy.deepcopy(info), agent_state)

            episode_metrics[agent.id]["reward"] += info["reward"]
            episode_metrics[agent.id]["episode_length"] += 1
            episode_metrics["full_episode_length"] += 1
        else:
            transition_info.start_agent(agent)

        stacked_observation = transition_info.get_stacked_state(agent, observation)
        action, agent_state = agent.act(stacked_observation, agent_state)
        (
            next_observation,
            reward,
            terminated,
            truncated,
            turn,
            other_info,
        ) = environment.step(action)
        transition_info.record_info(
            agent,
            {
                "observation": observation,
                "action": action,
                "info": other_info,
            },
        )
        if self._self_play:
            transition_info.record_info(
                agent,
                {
                    "agent_id": agent.id,
                },
            )
        transition_info.update_all_rewards(reward)
        agent_states[turn] = agent_state
        return terminated, truncated, next_observation, turn, agent_states

    def run_end_step(
        self,
        episode_metrics,
        transition_info,
        agent_states,
        terminated=True,
        truncated=False,
    ):
        """Run the final step of an episode.

        After an episode ends, iterate through agents and update then with the final
        step in the episode.

        Args:
            episode_metrics (Metrics): Keeps track of metrics for current episode.
            terminated (bool): Whether this step was terminal.
            truncated (bool): Whether this step was terminal.

        """
        for idx, agent in enumerate(self._agents):
            if transition_info.is_started(agent):
                info = transition_info.get_info(agent, terminated, truncated)
                agent_state = agent_states[idx]
                if self._training:
                    agent_state = agent.update(copy.deepcopy(info), agent_state)
                episode_metrics[agent.id]["episode_length"] += 1
                episode_metrics[agent.id]["reward"] += info["reward"]
                episode_metrics["full_episode_length"] += 1

    def run_episode(self, environment):
        """Run a single episode of the environment."""
        episode_metrics = self.create_episode_metrics()
        observation, turn = environment.reset()
        transition_info = TransitionInfo(self._agents, self._stack_size)
        steps = 0
        agent_states = [None] * len(self._agents)
        terminated, truncated = False, False

        # Run the loop until the episode ends or times out
        while (
            not (terminated or truncated)
            and steps < self._max_steps_per_episode
            and (not self._training or self._train_schedule.get_value())
        ):
            terminated, truncated, observation, turn, agent_states = self.run_one_step(
                environment,
                observation,
                turn,
                episode_metrics,
                transition_info,
                agent_states,
            )
            if self._run_testing and self._training:
                # Run test episodes
                self.run_testing()

            steps += 1
            if steps == self._max_steps_per_episode:
                truncated = not terminated

        # Run the final update.
        self.run_end_step(
            episode_metrics, transition_info, agent_states, terminated, truncated
        )
        return episode_metrics
