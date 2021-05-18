import pytest
import os

import numpy as np

from argparse import Namespace
from hive.runners.utils import load_config
from hive.runners import single_agent_loop

args = Namespace(
    config="hive/runners/tests/test_sa_config.yml",
    agent_config=None,
    env_config=None,
    logger_config=None,
)


@pytest.fixture()
def initial_runner():
    config = load_config(args)
    runner = single_agent_loop.set_up_experiment(config)

    return runner, config


def test_train_mode(initial_runner):
    """
    test setting the agents to train mode
    """
    single_agent_runner, _ = initial_runner
    single_agent_runner.train_mode(True)
    for agent in single_agent_runner._agents:
        assert agent._qnet.training is True
    single_agent_runner.train_mode(False)
    for agent in single_agent_runner._agents:
        assert agent._qnet.training is False


def test_run_step(initial_runner):
    """
    test running one step
    """
    single_agent_loop, config = initial_runner
    episode_metrics = single_agent_loop.create_episode_metrics()
    done = False
    observation, turn = single_agent_loop._environment.reset()
    assert turn == 0
    agent = single_agent_loop._agents[turn]
    done, observation, turn = single_agent_loop.run_one_step(
        observation, turn, episode_metrics
    )
    single_agent_loop._train_step_schedule.update()
    assert episode_metrics[agent.id]["episode_length"] == 1


def test_run_episode(initial_runner):
    """
    test running one episode
    """
    single_agent_runner, config = initial_runner
    episode_metrics = single_agent_runner.run_episode()
    agent = single_agent_runner._agents[0]
    assert (
        episode_metrics[agent._id]["episode_length"]
        == episode_metrics[agent._id]["reward"]
    )
    assert (
        episode_metrics[agent._id]["episode_length"]
        == single_agent_runner._train_step_schedule._steps
    )
    single_agent_runner._experiment_manager.save()


def test_resume(initial_runner):
    """
    test running training
    """
    resumed_single_agent_runner, config = initial_runner
    assert resumed_single_agent_runner._train_step_schedule._steps == 0

    resumed_single_agent_runner._experiment_manager.resume()
    resumed_single_agent_runner._train_step_schedule = (
        resumed_single_agent_runner._experiment_manager.experiment_state[
            "train_step_schedule"
        ]
    )
    resumed_single_agent_runner._train_episode_schedule = (
        resumed_single_agent_runner._experiment_manager.experiment_state[
            "train_episode_schedule"
        ]
    )
    resumed_single_agent_runner._test_schedule = (
        resumed_single_agent_runner._experiment_manager.experiment_state["test_schedule"]
    )
    episode_metrics = resumed_single_agent_runner.run_episode()


def test_run_training(initial_runner):
    """
    test running training
    """
    single_agent_runner, config = initial_runner
    single_agent_runner.run_training()
    assert single_agent_runner._train_step_schedule._steps == config["train_steps"]
