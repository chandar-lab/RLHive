import pytest
import os

import numpy as np
from hive import envs, replays
from argparse import Namespace
from hive.runners.utils import load_config
from hive.runners import multi_agent_loop, single_agent_loop

args = Namespace(config="./test_config.yml", agent_config=None, env_config=None, logger_config=None)


@pytest.fixture()
def initial_runner():
    config = load_config(args)
    runner = multi_agent_loop.set_up_experiment(config)

    return runner, config


def test_train_mode(initial_runner):
    """
        test setting the agents to train mode
    """
    multi_agent_runner, _ = initial_runner
    multi_agent_runner.train_mode(True)
    for agent in multi_agent_runner._agents:
        assert agent._qnet.training is True
    multi_agent_runner.train_mode(False)
    for agent in multi_agent_runner._agents:
        assert agent._qnet.training is False


def test_run_step(initial_runner):
    """
        test running one step
    """
    multi_agent_runner, config = initial_runner
    episode_metrics = multi_agent_runner.create_episode_metrics()
    done = False
    observation, turn = multi_agent_runner._environment.reset()
    assert turn == 0
    agent = multi_agent_runner._agents[turn]
    done, observation, turn = multi_agent_runner.run_one_step(
            observation, turn, episode_metrics
        )
    episode_metrics[agent.id]["episode_length"] += 1
    episode_metrics["full_episode_length"] += 1
    multi_agent_runner._train_step_schedule.update()
    assert episode_metrics[agent.id]["episode_length"] == 1


@pytest.mark.parametrize("max_steps_per_episode", [100])
def test_run_episode(initial_runner,
                     max_steps_per_episode):
    """
        test running one episode
    """
    multi_agent_runner, config = initial_runner
    episode_metrics = multi_agent_runner.run_episode()
    assert episode_metrics["full_episode_length"] == len(multi_agent_runner._agents) * max_steps_per_episode
    for agent in multi_agent_runner._agents:
        assert episode_metrics[agent._id]['episode_length'] == max_steps_per_episode
    multi_agent_runner._experiment_manager.save()


def test_run_training(initial_runner):
    """
        test running training
    """
    multi_agent_runner, config = initial_runner
    multi_agent_runner.run_training()
    assert multi_agent_runner._train_step_schedule._steps == config['train_steps']


@pytest.mark.parametrize("max_steps_per_episode", [100])
def test_resume(initial_runner, max_steps_per_episode):
    """
        test running training
    """
    resumed_multi_agent_runner, config = initial_runner
    assert resumed_multi_agent_runner._train_step_schedule._steps == 0

    resumed_multi_agent_runner._experiment_manager.resume()
    resumed_multi_agent_runner._train_step_schedule = resumed_multi_agent_runner._experiment_manager.experiment_state[
        "train_step_schedule"
    ]
    resumed_multi_agent_runner._train_episode_schedule = resumed_multi_agent_runner._experiment_manager.experiment_state[
        "train_episode_schedule"
    ]
    resumed_multi_agent_runner._test_schedule = resumed_multi_agent_runner._experiment_manager.experiment_state["test_schedule"]
    episode_metrics = resumed_multi_agent_runner.run_episode()
    assert resumed_multi_agent_runner._train_step_schedule._steps == 2 * len(resumed_multi_agent_runner._agents) * max_steps_per_episode