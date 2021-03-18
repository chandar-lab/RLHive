import argparse
import copy
import numpy as np
import torch
import yaml

from hive import agents, envs, replays
from hive.agents import qnets
from hive.utils import experiment, logging, schedule, utils


def run_single_agent_training(
    environment, agent, training_schedule, testing_schedule, logger, experiment_manager,
):
    agent.train()
    episode_metrics = {
        "cumulative_reward": 0,
        "episode_length": 0,
    }
    observation, _ = environment.reset()
    while training_schedule.update():
        done, observation = run_one_step(
            environment, agent, observation, episode_metrics
        )

        if done:
            if logger.update_step("train_episodes"):
                logger.log_metrics(episode_metrics, "train_episodes")
            episode_metrics = {
                "cumulative_reward": 0,
                "episode_length": 0,
            }
            while testing_schedule.update():
                # Run the testing loop for one episode
                agent.eval()
                done = False
                observation, _ = environment.reset()
                while not done:
                    done, observation = run_one_step(
                        environment, agent, observation, episode_metrics
                    )

                if logger.update_step("test_episodes"):
                    logger.log_metrics(episode_metrics, "test_episodes")

                # Reset the agent for training
                agent.train()
                episode_metrics = {
                    "cumulative_reward": 0,
                    "episode_length": 0,
                }
            observation, _ = environment.reset()
        if experiment_manager.update_step():
            experiment_manager.save()
    experiment_manager.save()


def run_one_step(environment, agent, observation, episode_metrics):
    action = agent.act(observation)
    next_observation, reward, done, turn, info = environment.step(action)
    agent.update(
        {
            "observation": observation,
            "action": action,
            "reward": reward,
            "next_observation": next_observation,
            "done": done,
            "turn": turn,
            "info": info,
        }
    )
    episode_metrics["cumulative_reward"] += reward
    episode_metrics["episode_length"] += 1
    return done, next_observation


def load_config(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.agent_config is not None:
        with open(args.agent_config) as f:
            config["agents"] = yaml.safe_load(f)
    if args.env_config is not None:
        with open(args.env_config) as f:
            config["environment"] = yaml.safe_load(f)
    if args.logger_config is not None:
        with open(args.logger_config) as f:
            config["loggers"] = yaml.safe_load(f)
    return config


def set_up_experiment(config):
    original_config = utils.Chomp(copy.deepcopy(config))
    environment = envs.get_env(config["environment"])
    env_spec = environment.env_spec

    logger_config = config.get("loggers", None)
    if logger_config is None or len(logger_config) == 0:
        logger = logging.NullLogger()
    elif len(logger_config) == 1:
        logger = logging.get_logger(logger_config[0])
    else:
        logger = logging.CompositeLogger(logger_config)

    config["agents"][0]["kwargs"]["env_spec"] = env_spec
    config["agents"][0]["kwargs"]["logger"] = logger
    agent = agents.get_agent(config["agents"][0])

    training_schedule = schedule.get_schedule(config["training_schedule"])
    testing_schedule = schedule.get_schedule(config["testing_schedule"])
    saving_schedule = schedule.get_schedule(config["saving_schedule"])
    state = utils.Chomp(
        {
            "training_schedule": training_schedule,
            "testing_schedule": testing_schedule,
            "saving_schedule": saving_schedule,
        }
    )

    experiment_manager = experiment.Experiment(
        config["run_name"], config["save_dir"], saving_schedule
    )
    experiment_manager.register_experiment(
        config=original_config, logger=logger, experiment_state=state, agents=agent,
    )
    if config.get("resume", False):
        experiment_manager.resume()
        training_schedule = state.training_schedule
        testing_schedule = state.testing_schedule
        saving_schedule = state.saving_schedule

    return (
        environment,
        agent,
        training_schedule,
        testing_schedule,
        logger,
        experiment_manager,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="./config.yaml")
    parser.add_argument("-a", "--agent-config")
    parser.add_argument("-e", "--env-config")
    parser.add_argument("-l", "--logger-config")
    args = parser.parse_args()
    config = load_config(args)
    (
        environment,
        agent,
        training_schedule,
        testing_schedule,
        logger,
        experiment_manager,
    ) = set_up_experiment(config)
    run_single_agent_training(
        environment,
        agent,
        training_schedule,
        testing_schedule,
        logger,
        experiment_manager,
    )

