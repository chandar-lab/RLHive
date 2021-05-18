import argparse
import copy
import numpy as np
import torch
import yaml

from hive import agents as agent_lib
from hive import envs
from hive.utils import experiment, logging, schedule, utils
from hive.runners.utils import load_config, TransitionInfo
from hive.runners.base import Runner


class MultiAgentRunner(Runner):
    """Runner class used to implement a multiagent training loop."""

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
        super().__init__(
            environment,
            agents,
            logger,
            experiment_manager,
            train_steps,
            train_episodes,
            test_frequency,
            test_num_episodes,
        )
        self._environment = environment
        self._agents = agents
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
            self._test_schedule = schedule.DoublePeriodicSchedule(
                False, True, test_frequency, test_num_episodes
            )
        self._train_step_schedule.update()
        self._test_schedule.update()
        self._experiment_manager.experiment_state.add_from_dict(
            {
                "train_step_schedule": self._train_step_schedule,
                "train_episode_schedule": self._train_episode_schedule,
                "test_schedule": self._test_schedule,
            }
        )

        self._transition_info = TransitionInfo(self._agents)

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
        agent = self._agents[turn]
        if self._transition_info.is_started(agent):
            info = self._transition_info.get_info(agent)
            agent.update(info)
            episode_metrics[agent.id]["reward"] += info["reward"]
            episode_metrics[agent.id]["episode_length"] += 1
            episode_metrics["full_episode_length"] += 1
        else:
            self._transition_info.start_agent(agent)
        action = agent.act(observation)
        next_observation, reward, done, turn, other_info = self._environment.step(
            action
        )
        self._transition_info.record_info(
            agent,
            {
                "observation": observation,
                "action": action,
                "next_observation": next_observation,
                "info": other_info,
            },
        )
        self._transition_info.update_all_rewards(reward)
        return done, next_observation, turn

    def run_end_step(self, episode_metrics):
        """Run the final step of an episode.

        After an episode ends, iterate through agents and update then with the final
        step in the episode.

        Args:
            episode_metrics: Metrics object keeping track of metrics for current episode.

        """
        for agent in self._agents:
            info = self._transition_info.get_info(agent, done=True)
            agent.update(info)
            episode_metrics[agent.id]["reward"] += info["reward"]
            episode_metrics[agent.id]["episode_length"] += 1
            episode_metrics["full_episode_length"] += 1

    def run_episode(self):
        """Run a single episode of the environment."""
        self._transition_info.reset()
        episode_metrics = self.create_episode_metrics()
        done = False
        observation, turn = self._environment.reset()

        # Run the loop until either training ends or the episode ends
        while self._train_step_schedule.get_value() and not done:
            done, observation, turn = self.run_one_step(
                observation, turn, episode_metrics
            )
            self._train_step_schedule.update()

        # If the episode ended, run the final update.
        if done:
            self.run_end_step(episode_metrics)
        return episode_metrics


def set_up_experiment(config):
    """Returns a runner object based on the config."""

    original_config = utils.Chomp(copy.deepcopy(config))

    # Set up environment
    environment = envs.get_env(config["environment"])
    env_spec = environment.env_spec

    # Set up loggers
    logger_config = config.get("loggers", None)
    if logger_config is None or len(logger_config) == 0:
        logger = logging.NullLogger()
    else:
        for logger in logger_config:
            logger["kwargs"]["timescales"] = ["train_episodes", "test_episodes"]
        if len(logger_config) == 1:
            logger = logging.get_logger(logger_config[0])
        else:
            logger = logging.CompositeLogger(logger_config)

    # Set up agents
    agents = []
    for idx, agent_config in enumerate(config["agents"]):
        agent_config["kwargs"]["obs_dim"] = env_spec.obs_dim[idx]
        agent_config["kwargs"]["act_dim"] = env_spec.act_dim[idx]
        agent_config["kwargs"]["logger"] = logger

        if not config["self_play"] or idx == 0:
            agents.append(agent_lib.get_agent(agent_config))
        else:
            agents.append(copy.copy(agents[0]))
            agents[-1]._id = idx

    # Set up experiment manager
    saving_schedule = schedule.get_schedule(config["saving_schedule"])
    experiment_manager = experiment.Experiment(
        config["run_name"], config["save_dir"], saving_schedule
    )
    experiment_manager.register_experiment(
        config=original_config,
        logger=logger,
        agents=agents,
    )

    # Set up runner
    runner = MultiAgentRunner(
        environment,
        agents,
        logger,
        experiment_manager,
        config.get("train_steps", -1),
        config.get("train_episodes", -1),
        config.get("test_frequency", -1),
        config.get("test_num_episodes", 1),
    )
    if config.get("resume", False):
        runner.resume()

    return runner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="./config.yml")
    parser.add_argument("-a", "--agent-config")
    parser.add_argument("-e", "--env-config")
    parser.add_argument("-l", "--logger-config")
    args = parser.parse_args()
    config = load_config(args)
    runner = set_up_experiment(config)
    runner.run_training()
