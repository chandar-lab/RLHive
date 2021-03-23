import argparse
import copy
import numpy as np
import torch
import yaml

from hive import agents, envs, replays
from hive.agents import qnets
from hive.utils import experiment, logging, schedule, utils


class Metrics:
    def __init__(self, agents, agent_metrics, episode_metrics):
        self._metrics = {}
        self._agent_metrics = agent_metrics
        self._episode_metrics = episode_metrics
        self._agent_ids = [agent.id for agent in agents]
        self.reset_metrics()

    def reset_metrics(self):
        for agent_id in self._agent_ids:
            self._metrics[agent.id] = {}
            for metric_name, metric_value in self._agent_metrics:
                self._metrics[agent.id][metric_name] = (
                    metric_value() if callable(metric_value) else metric_value
                )

        for metric_name, metric_value in self._episode_metrics:
            self._metrics[metric_name] = (
                metric_value() if callable(metric_value) else metric_value
            )

    def get_flat_dict(self):
        metrics = {}
        for metric, _ in self._episode_metrics:
            metrics[metric] = self._metrics[metric]
        for agent_id in self._agent_ids:
            for metric, _ in self._agent_metrics:
                metrics[f"{agent_id}_{metric}"] = self._metrics[agent_id][metric]
        return metrics

    def __getitem__(self, key):
        return self._metrics[key]

    def __setitem__(self, key, value):
        self._metrics[key] = value


def train_mode(agents, training):
    for agent in agents:
        agent.train() if training else agent.eval()


def run_multi_agent_training(
    environment,
    agents,
    training_schedule,
    testing_schedule,
    logger,
    experiment_manager,
):
    train_mode(agents, True)
    episode_metrics = Metrics(
        agents, [("reward", 0), ("episode_length", 0)], [("full_episode_length", 0)],
    )

    num_done = 0
    observation, turn = environment.reset()

    while training_schedule.update():
        done, observation, turn = run_one_step(
            environment, agents[turn], observation, episode_metrics
        )
        if done:
            num_done += 1
        if num_done == len(agents):
            if logger.update_step("train_episodes"):
                logger.log_metrics(episode_metrics.get_flat_dict(), "train_episodes")
            episode_metrics.reset_metrics()
            while testing_schedule.update():
                # Run the testing loop for one episode
                train_mode(agents, False)
                num_done = 0
                observation, turn = environment.reset()
                while num_done < len(agents):
                    done, observation, turn = run_one_step(
                        environment, agents[turn], observation, episode_metrics
                    )
                    if done:
                        num_done += 1

                if logger.update_step("test_episodes"):
                    logger.log_metrics(episode_metrics.get_flat_dict(), "test_episodes")

                # Reset the agent for training
                train_mode(agents, True)
                episode_metrics.reset_metrics()
            num_done = 0
            observation, turn = environment.reset()
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
            "info": info,
        }
    )
    episode_metrics[agent.id]["reward"] += reward
    episode_metrics[agent.id]["episode_length"] += 1
    episode_metrics["full_episode_length"] += 1
    return done, next_observation, turn


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
    else:
        for logger in logger_config:
            logger["kwargs"]["timescales"] = ["train_episodes", "test_episodes"]
        if len(logger_config) == 1:
            logger = logging.get_logger(logger_config[0])
        else:
            logger = logging.CompositeLogger(logger_config)

    agents = []
    for agent_config in config["agents"]:
        agent_config["kwargs"]["env_spec"] = env_spec
        agent_config["kwargs"]["logger"] = logger
        agents.append(agents.get_agent(agent_config))

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
        config=original_config, logger=logger, experiment_state=state, agents=agents,
    )
    if config.get("resume", False):
        experiment_manager.resume()
        training_schedule = state.training_schedule
        testing_schedule = state.testing_schedule
        saving_schedule = state.saving_schedule

    return (
        environment,
        agents,
        training_schedule,
        testing_schedule,
        logger,
        experiment_manager,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="./config.yml")
    parser.add_argument("-a", "--agent-config")
    parser.add_argument("-e", "--env-config")
    parser.add_argument("-l", "--logger-config")
    args = parser.parse_args()
    config = load_config(args)
    (
        environment,
        agents,
        training_schedule,
        testing_schedule,
        logger,
        experiment_manager,
    ) = set_up_experiment(config)
    run_single_agent_training(
        environment,
        agents,
        training_schedule,
        testing_schedule,
        logger,
        experiment_manager,
    )

