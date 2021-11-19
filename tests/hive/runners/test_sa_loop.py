import os
import sys
from argparse import Namespace
from unittest.mock import patch

import hive
import pytest
from hive import runners
from hive.runners import single_agent_loop
from hive.runners.utils import load_config
from hive.utils.logging import Logger, ScheduledLogger
from hive.utils.schedule import ConstantSchedule


class FakeLogger1(ScheduledLogger):
    def __init__(self, timescales=None, logger_schedules=None, arg1: int = 0):
        super().__init__(timescales, logger_schedules)
        self.arg1 = arg1

    def log_config(self, config):
        pass

    def log_scalar(self, name, value, prefix):
        pass

    def log_metrics(self, metrics, prefix):
        pass

    def save(self, dir_name):
        pass

    def load(self, dir_name):
        pass


class FakeLogger2(ScheduledLogger):
    def __init__(self, timescales=None, logger_schedules=None, arg2: float = 0):
        super().__init__(timescales, logger_schedules)
        self.arg2 = arg2

    def log_config(self, config):
        pass

    def log_scalar(self, name, value, prefix):
        pass

    def log_metrics(self, metrics, prefix):
        pass

    def save(self, dir_name):
        pass

    def load(self, dir_name):
        pass


hive.registry.register("FakeLogger1", FakeLogger1, FakeLogger1)
hive.registry.register("FakeLogger2", FakeLogger2, FakeLogger2)


@pytest.fixture()
def args():
    return Namespace(
        config="tests/hive/runners/test_sa_config.yml",
        agent_config=None,
        env_config=None,
        logger_config=None,
    )


@pytest.fixture()
def initial_runner(args, tmpdir):
    config = load_config(args)
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
    done, observation = single_agent_loop.run_one_step(observation, episode_metrics)
    single_agent_loop._train_schedule.update()
    assert episode_metrics[agent.id]["episode_length"] == 1


def test_run_episode(initial_runner):
    """
    test running one episode
    """
    single_agent_runner, config = initial_runner
    episode_metrics, steps = single_agent_runner.run_episode()
    agent = single_agent_runner._agents[0]
    assert (
        episode_metrics[agent._id]["episode_length"]
        == episode_metrics[agent._id]["reward"]
    )
    assert (
        episode_metrics[agent._id]["episode_length"]
        == single_agent_runner._train_schedule._steps
    )
    assert steps == single_agent_runner._train_schedule._steps


def test_resume(initial_runner):
    """
    test running training
    """
    resumed_single_agent_runner, config = initial_runner
    assert resumed_single_agent_runner._train_schedule._steps == 0

    resumed_single_agent_runner._experiment_manager.resume()
    resumed_single_agent_runner._train_schedule = (
        resumed_single_agent_runner._experiment_manager.experiment_state[
            "train_schedule"
        ]
    )
    resumed_single_agent_runner._test_schedule = (
        resumed_single_agent_runner._experiment_manager.experiment_state[
            "test_schedule"
        ]
    )
    episode_metrics = resumed_single_agent_runner.run_episode()


def test_run_training(initial_runner):
    """
    test running training
    """
    single_agent_runner, config = initial_runner
    single_agent_runner.run_training()
    assert single_agent_runner._train_schedule._steps >= config["train_steps"]
    assert (
        single_agent_runner._train_schedule._steps
        <= config["train_steps"]
        + single_agent_runner._environment._env._max_episode_steps
    )


@pytest.mark.parametrize(
    "arg_string,cl_args",
    [
        (
            "single_agent_loop.py"
            " --agent.representation_net.hidden_units [30,30]"
            " --agent.discount_rate .8 "
            " --seed 20"
            " --loggers.logger_list.0.arg1 2"
            " --loggers.logger_list.1.arg2 .2",
            [[30, 30], 0.8, 20, 2, 0.2],
        ),
        (
            "single_agent_loop.py"
            " --loggers.logger_list.0.arg1 2"
            " --loggers.logger_list.1.arg2 .2",
            [None, None, None, 2, 0.2],
        ),
        (
            "single_agent_loop.py"
            " --agent.representation_net.hidden_units [30,30]"
            " --agent.discount_rate .8 ",
            [[30, 30], 0.8, None, None, None],
        ),
        (
            "single_agent_loop.py --seed 20",
            [None, None, 20, None, None],
        ),
    ],
)
@patch("hive.runners.single_agent_loop.set_seed")
def test_cl_parsing(mock_seed, args, arg_string, cl_args):
    defaults = [[256, 256], 0.99, None, 0, 0.5]
    expected_args = [
        cl_args[idx] if cl_args[idx] else defaults[idx] for idx in range(len(cl_args))
    ]
    sys.argv = arg_string.split()
    config = load_config(args)
    runner = single_agent_loop.set_up_experiment(config)
    full_config = runner._experiment_manager._config
    # Check hidden units
    assert (
        runner._agents[0]._qnet.network.network[0].out_features == expected_args[0][0]
    )
    assert (
        full_config["agent"]["kwargs"]["representation_net"]["kwargs"]["hidden_units"]
        == expected_args[0]
    )
    # Check discount factor
    assert runner._agents[0]._discount_rate == expected_args[1]
    assert full_config["agent"]["kwargs"]["discount_rate"] == expected_args[1]
    # Check Logger 1 arg
    assert runner._logger._logger_list[0].arg1 == expected_args[3]
    if cl_args[3]:
        assert (
            full_config["loggers"]["kwargs"]["logger_list"][0]["kwargs"]["arg1"]
            == expected_args[3]
        )
    else:
        assert (
            "arg1" not in full_config["loggers"]["kwargs"]["logger_list"][0]["kwargs"]
        )
    # Check Logger 2 arg
    assert runner._logger._logger_list[1].arg2 == expected_args[4]
    if cl_args[4]:
        assert (
            full_config["loggers"]["kwargs"]["logger_list"][1]["kwargs"]["arg2"]
            == expected_args[4]
        )

    # Check seed
    if cl_args[2]:
        assert mock_seed.call_args.args == (cl_args[2],)
    else:
        assert not mock_seed.called
