import argparse
import copy
import numpy as np
import torch
import yaml

from hive import agents as agent_lib
from hive import envs, replays
from hive.utils import experiment, logging, schedule, utils
from hive.runners.utils import load_config, Metrics, TransitionInfo


class Runner:
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

    def train_mode(self, training):
        for agent in self._agents:
            agent.train() if training else agent.eval()

    def create_episode_metrics(self):
        return Metrics(
            self._agents,
            [("reward", 0), ("episode_length", 0)],
            [("full_episode_length", 0)],
        )

    def run_one_step(self, observation, turn, episode_metrics):
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
        for agent in self._agents:
            info = self._transition_info.get_info(agent, done=True)
            agent.update(info)
            episode_metrics[agent.id]["reward"] += info["reward"]
            episode_metrics[agent.id]["episode_length"] += 1
            episode_metrics["full_episode_length"] += 1

    def run_episode(self):
        self._transition_info.reset()
        episode_metrics = self.create_episode_metrics()
        done = False
        observation, turn = self._environment.reset()
        while self._train_step_schedule.get_value() and not done:
            done, observation, turn = self.run_one_step(
                observation, turn, episode_metrics
            )
            self._train_step_schedule.update()
        if done:
            self.run_end_step(episode_metrics)
        return episode_metrics

    def run_training(self):
        self.train_mode(True)
        while (
            self._train_episode_schedule.update()
            and self._train_step_schedule.get_value()
        ):
            self.train_mode(True)
            episode_metrics = self.run_episode()
            if self._logger.update_step("train_episodes"):
                self._logger.log_metrics(
                    episode_metrics.get_flat_dict(), "train_episodes"
                )
            while self._test_schedule.update():
                self.train_mode(False)
                episode_metrics = self.run_episode()
                if self._logger.update_step("test_episodes"):
                    self._logger.log_metrics(
                        episode_metrics.get_flat_dict(), "test_episodes"
                    )
            if self._experiment_manager.update_step():
                self._experiment_manager.save()

        self.train_mode(False)
        episode_metrics = self.run_episode()
        self._logger.update_step("test_episodes")
        self._logger.log_metrics(episode_metrics.get_flat_dict(), "test_episodes")
        self._experiment_manager.save()

    def resume(self):
        self._experiment_manager.resume()
        self._train_step_schedule = self._experiment_manager.experiment_state[
            "train_step_schedule"
        ]
        self._train_episode_schedule = self._experiment_manager.experiment_state[
            "train_episode_schedule"
        ]
        self._test_schedule = self._experiment_manager.experiment_state["test_schedule"]


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
    for idx, agent_config in enumerate(config["agents"]):
        agent_config["kwargs"]["obs_dim"] = env_spec.obs_dim[idx]
        agent_config["kwargs"]["act_dim"] = env_spec.act_dim[idx]
        agent_config["kwargs"]["logger"] = logger
        agents.append(agent_lib.get_agent(agent_config))

    saving_schedule = schedule.get_schedule(config["saving_schedule"])

    experiment_manager = experiment.Experiment(
        config["run_name"], config["save_dir"], saving_schedule
    )
    experiment_manager.register_experiment(
        config=original_config, logger=logger, agents=agents,
    )
    runner = Runner(
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
