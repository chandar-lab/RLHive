import os
import sys
from argparse import Namespace
from unittest.mock import patch

import pytest

import hive
from hive.runners import single_agent_loop
from hive.runners.utils import load_config
from hive.utils.loggers import ScheduledLogger
from hive.runners.utils import TransitionInfo

@pytest.fixture()
def args():
    return Namespace(
        config="tests/hive/utils/metrics/test_metrics_config.yml",
        agent_config=None,
        env_config=None,
        logger_config=None,
    )

@pytest.fixture()
def initial_runner(args, tmpdir):
    config = load_config(
        args.config,
        agent_config=args.agent_config,
        env_config=args.env_config,
        logger_config=args.logger_config,
    )
    config["save_dir"] = os.path.join(tmpdir, config["save_dir"])
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

def test_reset(initial_runner):
    """
    test running one step
    """
    single_agent_loop, config = initial_runner
    episode_metrics = single_agent_loop.create_episode_metrics()
    agents = single_agent_loop._agents
    stack_size = 5
    t_info = TransitionInfo(agents, stack_size)
    assert t_info._transitions == {agent_id: {"reward": 0.0} for agent_id in t_info._agent_ids}
    assert t_info._started == {agent_id: False for agent_id in t_info._agent_ids}

def test_is_started(initial_runner):
    single_agent_loop, config = initial_runner
    episode_metrics = single_agent_loop.create_episode_metrics()
    agents = single_agent_loop._agents
    stack_size = 5
    t_info = TransitionInfo(agents, stack_size)
    assert t_info.is_started(agents[0]) == False

def test_start_agent(initial_runner):
    single_agent_loop, config = initial_runner
    episode_metrics = single_agent_loop.create_episode_metrics()
    agents = single_agent_loop._agents
    stack_size = 5
    t_info = TransitionInfo(agents, stack_size)
    t_info.start_agent(agents[0])
    assert t_info.is_started (agents[0]) == True

def test_update_reward(initial_runner):
    single_agent_loop, config = initial_runner
    episode_metrics = single_agent_loop.create_episode_metrics()
    agents = single_agent_loop._agents
    stack_size = 5
    t_info = TransitionInfo(agents, stack_size)
    assert t_info.update_reward(t_info._agent_ids[0], 1.)._transitions[t_info._agent_ids[0]]["reward"] == 1.

def test_record_info(initial_runner):
    pass