import argparse
import copy
import numpy as np
import torch
import yaml

from hive import agents as agent_lib
from hive import envs
from hive.utils import experiment, logging, schedule, utils
from hive.runners.utils import load_config
from hive.runners.base import Runner


class SingleAgentRunner(Runner):
    """Runner class used to implement a sinle-agent training loop."""

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
        action = agent.act(observation)
        next_observation, reward, done, _, other_info = self._environment.step(action)

        info = {
            "observation": observation,
            "reward": reward,
            "action": action,
            "next_observation": next_observation,
            "done": done,
            "info": other_info,
        }
        agent.update(info)
        episode_metrics[agent.id]["reward"] += info["reward"]
        episode_metrics[agent.id]["episode_length"] += 1
        episode_metrics["full_episode_length"] += 1

        return done, next_observation, _

    def run_episode(self):
        """Run a single episode of the environment."""
        episode_metrics = self.create_episode_metrics()
        done = False
        observation, _ = self._environment.reset()

        # Run the loop until either training ends or the episode ends
        while self._train_step_schedule.get_value() and not done:
            done, observation, _ = self.run_one_step(observation, 0, episode_metrics)
            self._train_step_schedule.update()

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

    # Set up agent
    config["agent"]["kwargs"]["obs_dim"] = env_spec.obs_dim[0]
    config["agent"]["kwargs"]["act_dim"] = env_spec.act_dim[0]
    config["agent"]["kwargs"]["logger"] = logger

    if "replay_buffer" in config["agent"]["kwargs"]:
        replay_args = config["agent"]["kwargs"]["replay_buffer"]["kwargs"]
        replay_args["observation_shape"] = env_spec.obs_dim[0]

    agent = agent_lib.get_agent(config["agent"])

    # Set up experiment manager
    saving_schedule = schedule.get_schedule(config["saving_schedule"])
    experiment_manager = experiment.Experiment(
        config["run_name"], config["save_dir"], saving_schedule
    )
    experiment_manager.register_experiment(
        config=original_config,
        logger=logger,
        agents=agent,
    )

    # Set up runner
    runner = SingleAgentRunner(
        environment,
        agent,
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
