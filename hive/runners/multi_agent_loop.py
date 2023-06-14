import copy
from typing import Optional, Sequence, Union

from hive.agents.agent import Agent
from hive.envs.base import BaseEnv
from hive.runners.base import Runner
from hive.runners.utils import TransitionInfo
from hive.types import Creates, default
from hive.utils import utils
from hive.utils.experiment import Experiment
from hive.utils.loggers import Logger


class MultiAgentRunner(Runner):
    """Runner class used to implement a multiagent training loop."""

    def __init__(
        self,
        environment: Creates[BaseEnv],
        agents: Sequence[Creates[Agent]],
        loggers: Optional[Union[Creates[Logger], Sequence[Creates[Logger]]]],
        experiment_manager: Creates[Experiment],
        train_steps: int,
        num_agents: int,
        eval_environment: Optional[Creates[BaseEnv]] = None,
        test_frequency: int = -1,
        test_episodes: int = 1,
        self_play: bool = False,
        max_steps_per_episode: int = 1_000_000_000,
        seed: Optional[int] = None,
    ):
        """Initializes the MultiAgentRunner object.

        Args:
            environment (BaseEnv): Environment used in the training loop.
            agent (Agent): Agent that will interact with the environment
            loggers (List[ScheduledLogger]): List of loggers used to log metrics.
            experiment_manager (Experiment): Experiment object that saves the state of
                the training.
            train_steps (int): How many steps to train for. This is the number
                of times that agent.update is called. If this is -1, there is no
                limit for the number of training steps.
            num_agents (int): Number of agents running in this multiagent experiment.
            eval_environment (BaseEnv): Environment used to evaluate the agent. If
                None, the ``environment`` parameter (which is a function) is
                used to create a second environment.
            test_frequency (int): After how many training steps to run testing
                episodes. If this is -1, testing is not run.
            test_episodes (int): How many episodes to run testing for duing each test
                phase.
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
        eval_environment = default(eval_environment, environment)
        created_environment = environment()
        created_eval_environment = eval_environment() if test_frequency != -1 else None
        env_spec = created_environment.env_spec

        agent_list = []
        num_agents = num_agents if self_play else len(agents)
        for idx in range(num_agents):
            if not self_play or idx == 0:
                agent_fn = agents[idx]
                agent = agent_fn(
                    observation_space=env_spec.observation_space[idx],
                    action_space=env_spec.action_space[idx],
                )
                agent_list.append(agent)
            else:
                agent_list.append(copy.copy(agent_list[0]))
                agent_list[-1]._id = f"{agent_list[0]._id}_{idx}"
        if self_play:
            agent_list[0]._id = f"{agent_list[0]._id}_{0}"
        # Set up experiment manager

        super().__init__(
            environment=created_environment,
            eval_environment=created_eval_environment,
            agents=agent_list,
            loggers=loggers,
            experiment_manager=experiment_manager(),
            train_steps=train_steps,
            test_frequency=test_frequency,
            test_episodes=test_episodes,
            max_steps_per_episode=max_steps_per_episode,
        )
        self._self_play = self_play

    def run_one_step(
        self,
        environment,
        observation,
        turn,
        episode_metrics,
        transition_info,
        agent_traj_states,
    ):
        """Run one step of the training loop.

        If it is the agent's first turn during the episode, do not run an update step.
        Otherwise, run an update step based on the previous action and accumulated
        reward since then.

        Args:
            environment (BaseEnv): Environment in which the agent will take a step in.
            observation: Current observation that the agent should create an action
                for.
            turn (int): Agent whose turn it is.
            episode_metrics (Metrics): Keeps track of metrics for current
                episode.
            transition_info (TransitionInfo): Used to keep track of the most
                recent transition for each agent.
            agent_traj_states: List of trajectory state objects that will be
                passed to each agent when act and update are called. The agent
                returns new trajectory states to replace the state passed in.
        """
        agent = self._agents[turn]
        agent_traj_state = agent_traj_states[turn]
        if transition_info.is_started(agent):
            info = transition_info.get_info(agent)
            if self._training:
                agent_traj_state = agent.update(
                    copy.deepcopy(info), agent_traj_state, self._train_steps
                )
            episode_metrics[agent.id]["reward"] += info["reward"]
            episode_metrics[agent.id]["episode_length"] += 1
            episode_metrics["full_episode_length"] += 1
        else:
            transition_info.start_agent(agent)

        action, agent_traj_state = agent.act(
            observation, agent_traj_state, self._train_steps
        )
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
                "next_observation": next_observation,
                "action": action,
                "info": other_info,
                "source": agent.id,
            },
        )
        transition_info.update_all_rewards(reward)
        agent_traj_states[turn] = agent_traj_state
        return terminated, truncated, next_observation, turn, agent_traj_states

    def run_end_step(
        self,
        episode_metrics,
        transition_info,
        agent_traj_states,
        terminated=True,
        truncated=False,
    ):
        """Run the final step of an episode.

        After an episode ends, iterate through agents and update then with the final
        step in the episode.

        Args:
            episode_metrics (Metrics): Keeps track of metrics for current episode.
            transition_info (TransitionInfo): Used to keep track of the most
                recent transition for each agent.
            agent_traj_states: List of trajectory state objects that will be
                passed to each agent when act and update are called. The agent
                returns new trajectory states to replace the state passed in.
            terminated (bool): Whether this step was terminal.
            truncated (bool): Whether this step was terminal.
        """
        for idx, agent in enumerate(self._agents):
            if transition_info.is_started(agent):
                info = transition_info.get_info(agent, terminated, truncated)
                agent_traj_state = agent_traj_states[idx]
                if self._training:
                    agent_traj_state = agent.update(
                        copy.deepcopy(info), agent_traj_state, self._train_steps
                    )
                episode_metrics[agent.id]["episode_length"] += 1
                episode_metrics[agent.id]["reward"] += info["reward"]
                episode_metrics["full_episode_length"] += 1

    def run_episode(self, environment):
        """Run a single episode of the environment.

        Args:
            environment (BaseEnv): Environment in which the agent will take a step in.
        """
        episode_metrics = self.create_episode_metrics()
        observation, turn = environment.reset()
        transition_info = TransitionInfo(self._agents)
        steps = 0
        agent_traj_states = [None] * len(self._agents)
        terminated, truncated = False, False

        # Run the loop until the episode ends or times out
        while (
            not (terminated or truncated)
            and steps < self._max_steps_per_episode
            and (not self._training or self._train_schedule(self._train_steps))
        ):
            (
                terminated,
                truncated,
                observation,
                turn,
                agent_traj_states,
            ) = self.run_one_step(
                environment,
                observation,
                turn,
                episode_metrics,
                transition_info,
                agent_traj_states,
            )

            self.update_step()
            steps += 1
            if steps == self._max_steps_per_episode:
                truncated = not terminated

        if self._train_schedule(self._train_steps):
            # Run the final update.
            self.run_end_step(
                episode_metrics,
                transition_info,
                agent_traj_states,
                terminated,
                truncated,
            )
            self.update_step()

        return episode_metrics
