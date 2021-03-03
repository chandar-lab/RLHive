import argparse

import gym
import numpy as np
import torch
from hive import agents, envs
from hive.agents import qnets
from hive.utils import schedule, logging


def run_single_agent_training(environment, agent, logger, training_schedule):
    obs_0 = environment.reset()
    cum_reward = 0
    while training_schedule.update():
        action = agent.act(obs_0, training=True)
        obs_1, reward, done, info = environment.step(action)
        agent.update(
            {
                "state_0": obs_0,
                "action": action,
                "reward": reward,
                "state_1": obs_1,
                "done": done,
                "info": info,
            }
        )
        cum_reward += reward

        if done:
            logger.update_step()
            if logger.should_log():
                logger.log_scalar("cum_reward", cum_reward)
            cum_reward = 0
            obs_0 = environment.reset()
        else:
            obs_0 = obs_1


def set_up_experiment():
    # This is v1 hardcoded experiment. Should be initialized using command line arguments
    environment = gym.make("CartPole-v1")
    # environment = gym.make("MountainCar-v0")
    env_spec = envs.EnvSpec(
        "MountainCar-v0",
        environment.observation_space.shape[0],
        environment.action_space.n,
    )
    agent_logger = logging.WandbLogger(
        "Hive-v1",
        "test-run",
        logger_schedule=schedule.PeriodicSchedule(False, True, 50),
        logger_name="agent",
    )
    agent = agents.DQNAgent(
        qnets.SimpleMLP(env_spec), env_spec, torch.optim.Adam, logger=agent_logger
    )
    training_schedule = schedule.SwitchSchedule(True, False, 100000)
    logger = logging.WandbLogger(
        "Hive-v1",
        "test-run",
        logger_schedule=schedule.ConstantSchedule(True),
        logger_name="runner",
    )
    return environment, agent, training_schedule, logger


if __name__ == "__main__":
    environment, agent, training_schedule, logger = set_up_experiment()
    run_single_agent_training(environment, agent, logger, training_schedule)
