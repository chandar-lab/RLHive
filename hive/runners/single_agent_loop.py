import argparse
import copy
import numpy as np
import torch
import yaml

from hive import agents as agent_lib
from hive import envs, replays
from hive.utils import experiment, logging, schedule, utils
from hive.runners.utils import load_config, Metrics, TransitionInfo
from hive.runners.base import SingleAgentRunner


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
    agent = agent_lib.get_agent(config["agent"])

    # Set up experiment manager
    saving_schedule = schedule.get_schedule(config["saving_schedule"])
    experiment_manager = experiment.Experiment(
        config["run_name"], config["save_dir"], saving_schedule
    )
    experiment_manager.register_experiment(
        config=original_config, logger=logger, agents=agent,
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
