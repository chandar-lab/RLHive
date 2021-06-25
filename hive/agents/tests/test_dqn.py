from copy import deepcopy
from hive.agents.qnets.base import FunctionApproximator
from unittest.mock import Mock
import numpy as np
import pytest
import torch

from hive.agents import DQNAgent, RainbowDQNAgent, get_agent
from hive.agents.qnets import SimpleMLP, ComplexMLP, DistributionalMLP
from hive.envs import EnvSpec
from hive.replays import CircularReplayBuffer
from hive.utils import schedule
from torch.optim import Adam


@pytest.fixture
def env_spec():
    return EnvSpec("test_env", (2,), 2)


@pytest.fixture
def env_spec():
    return EnvSpec("test_env", (2,), 2)


"""
ddnd = double, dueling, noisy, distributional. x = False.
"""


@pytest.fixture(
    params=[
        pytest.lazy_fixture("xxxx_agent_with_mock_optimizer"),
        pytest.lazy_fixture("dxxx_agent_with_mock_optimizer"),
        pytest.lazy_fixture("xdxx_agent_with_mock_optimizer"),
    ]
)
def agent_with_mock_optimizer(request):
    return request.param


@pytest.fixture
def ddnd_agent_with_mock_optimizer(env_spec):
    supports = torch.linspace(0, 200, 51).to("cpu")
    agent = RainbowDQNAgent(
        qnet=FunctionApproximator(DistributionalMLP)(
            supports=supports,
            hidden_units=5,
            num_hidden_layers=1,
            noisy=True,
            dueling=True,
        ),
        obs_dim=env_spec.obs_dim,
        act_dim=env_spec.act_dim,
        optimizer_fn=Mock(),
        replay_buffer=CircularReplayBuffer(capacity=10),
        target_net_update_fraction=0.25,
        target_net_soft_update=True,
        target_net_update_schedule=schedule.PeriodicSchedule(False, True, 5),
        epsilon_schedule=schedule.LinearSchedule(1.0, 0.1, 20),
        learn_schedule=schedule.SwitchSchedule(False, True, 2),
        device="cpu",
        batch_size=2,
        use_eps_greedy=False,
        distributional=True,
        double=True,
    )
    return agent


@pytest.fixture
def dxxx_agent_with_mock_optimizer(env_spec):
    agent = RainbowDQNAgent(
        qnet=FunctionApproximator(ComplexMLP)(hidden_units=5, num_hidden_layers=1),
        obs_dim=env_spec.obs_dim,
        act_dim=env_spec.act_dim,
        optimizer_fn=Mock(),
        replay_buffer=CircularReplayBuffer(capacity=10),
        target_net_update_fraction=0.25,
        target_net_soft_update=True,
        target_net_update_schedule=schedule.PeriodicSchedule(False, True, 5),
        epsilon_schedule=schedule.LinearSchedule(1.0, 0.1, 20),
        learn_schedule=schedule.SwitchSchedule(False, True, 2),
        device="cpu",
        batch_size=2,
        double=True,
        distributional=False,
    )
    return agent


@pytest.fixture
def xdxx_agent_with_mock_optimizer(env_spec):
    agent = RainbowDQNAgent(
        qnet=FunctionApproximator(ComplexMLP)(
            hidden_units=5,
            num_hidden_layers=1,
            dueling=True,
        ),
        obs_dim=env_spec.obs_dim,
        act_dim=env_spec.act_dim,
        optimizer_fn=Mock(),
        replay_buffer=CircularReplayBuffer(capacity=10),
        target_net_update_fraction=0.25,
        target_net_soft_update=True,
        target_net_update_schedule=schedule.PeriodicSchedule(False, True, 5),
        epsilon_schedule=schedule.LinearSchedule(1.0, 0.1, 20),
        learn_schedule=schedule.SwitchSchedule(False, True, 2),
        device="cpu",
        batch_size=2,
        distributional=False,
    )
    return agent


@pytest.fixture
def xxnx_agent_with_mock_optimizer(env_spec):
    agent = RainbowDQNAgent(
        qnet=FunctionApproximator(ComplexMLP)(
            hidden_units=5,
            num_hidden_layers=1,
            noisy=True,
        ),
        obs_dim=env_spec.obs_dim,
        act_dim=env_spec.act_dim,
        optimizer_fn=Mock(),
        replay_buffer=CircularReplayBuffer(capacity=10),
        target_net_update_fraction=0.25,
        target_net_soft_update=True,
        target_net_update_schedule=schedule.PeriodicSchedule(False, True, 5),
        epsilon_schedule=schedule.LinearSchedule(1.0, 0.1, 20),
        learn_schedule=schedule.SwitchSchedule(False, True, 2),
        device="cpu",
        batch_size=2,
        use_eps_greedy=False,
        distributional=False,
    )
    return agent


@pytest.fixture
def xxxd_agent_with_mock_optimizer(env_spec):
    supports = torch.linspace(0, 200, 51).to("cpu")
    agent = RainbowDQNAgent(
        qnet=FunctionApproximator(DistributionalMLP)(
            supports=supports,
            hidden_units=5,
            num_hidden_layers=1,
        ),
        obs_dim=env_spec.obs_dim,
        act_dim=env_spec.act_dim,
        optimizer_fn=Mock(),
        replay_buffer=CircularReplayBuffer(capacity=10),
        target_net_update_fraction=0.25,
        target_net_soft_update=True,
        target_net_update_schedule=schedule.PeriodicSchedule(False, True, 5),
        epsilon_schedule=schedule.LinearSchedule(1.0, 0.1, 20),
        learn_schedule=schedule.SwitchSchedule(False, True, 2),
        device="cpu",
        batch_size=2,
        distributional=True,
    )
    return agent


@pytest.fixture
def xxxx_agent_with_mock_optimizer(env_spec):
    agent = DQNAgent(
        qnet=FunctionApproximator(SimpleMLP)(hidden_units=5, num_hidden_layers=1),
        obs_dim=env_spec.obs_dim,
        act_dim=env_spec.act_dim,
        optimizer_fn=Mock(),
        replay_buffer=CircularReplayBuffer(capacity=10),
        target_net_update_fraction=0.25,
        target_net_soft_update=True,
        target_net_update_schedule=schedule.PeriodicSchedule(False, True, 5),
        epsilon_schedule=schedule.LinearSchedule(1.0, 0.1, 20),
        learn_schedule=schedule.SwitchSchedule(False, True, 2),
        device="cpu",
        batch_size=2,
    )
    return agent


@pytest.fixture
def xxxx_rainbow_agent_with_mock_optimizer(env_spec):
    agent = RainbowDQNAgent(
        qnet=FunctionApproximator(ComplexMLP)(
            hidden_units=5,
            num_hidden_layers=1,
            noisy=False,
            dueling=False,
        ),
        obs_dim=env_spec.obs_dim,
        act_dim=env_spec.act_dim,
        optimizer_fn=Mock(),
        replay_buffer=CircularReplayBuffer(capacity=10),
        target_net_update_fraction=0.25,
        target_net_soft_update=True,
        target_net_update_schedule=schedule.PeriodicSchedule(False, True, 5),
        epsilon_schedule=schedule.LinearSchedule(1.0, 0.1, 20),
        learn_schedule=schedule.SwitchSchedule(False, True, 2),
        device="cpu",
        batch_size=2,
        use_eps_greedy=False,
        distributional=False,
        double=False,
    )
    return agent


@pytest.fixture
def agent_with_optimizer(env_spec):
    agent = DQNAgent(
        qnet=FunctionApproximator(SimpleMLP)(hidden_units=5, num_hidden_layers=1),
        obs_dim=env_spec.obs_dim,
        act_dim=env_spec.act_dim,
        optimizer_fn=Adam,
        replay_buffer=CircularReplayBuffer(capacity=10),
        target_net_update_fraction=0.25,
        target_net_soft_update=True,
        target_net_update_schedule=schedule.PeriodicSchedule(False, True, 5),
        epsilon_schedule=schedule.LinearSchedule(1.0, 0.1, 20),
        learn_schedule=schedule.SwitchSchedule(False, True, 2),
        device="cpu",
        batch_size=2,
    )
    return agent


def test_create_agent_with_configs(env_spec):
    agent_config = {
        "name": "DQNAgent",
        "kwargs": {
            "qnet": {
                "name": "SimpleMLP",
                "kwargs": {"hidden_units": 5, "num_hidden_layers": 1},
            },
            "obs_dim": env_spec.obs_dim,
            "act_dim": env_spec.act_dim,
            "optimizer_fn": {"name": "Adam", "kwargs": {"lr": 0.01}},
            "replay_buffer": {
                "name": "CircularReplayBuffer",
                "kwargs": {"capacity": 10},
            },
            "target_net_update_schedule": {
                "name": "PeriodicSchedule",
                "kwargs": {"off_value": False, "on_value": True, "period": 5},
            },
            "epsilon_schedule": {
                "name": "LinearSchedule",
                "kwargs": {"init_value": 1.0, "end_value": 0.1, "steps": 20},
            },
            "learn_schedule": {
                "name": "SwitchSchedule",
                "kwargs": {"off_value": False, "on_value": True, "steps": 2},
            },
            "device": "cpu",
        },
    }
    agent = get_agent(agent_config)
    action = agent.act(np.zeros(2))
    assert action < 2


def test_train_step(agent_with_mock_optimizer):
    agent_with_mock_optimizer.train()
    for idx in range(8):
        observation = np.ones(2) * (idx + 1)
        action = agent_with_mock_optimizer.act(observation)
        assert action < 2
        agent_with_mock_optimizer.update(
            {
                "action": action,
                "observation": observation,
                "reward": 1,
                "done": False,
            }
        )
    assert agent_with_mock_optimizer._optimizer.step.call_count == 7
    assert agent_with_mock_optimizer._replay_buffer.size() == 7
    assert agent_with_mock_optimizer._epsilon_schedule._value == pytest.approx(0.775)


def test_eval_step(agent_with_mock_optimizer):
    agent_with_mock_optimizer.eval()
    for idx in range(8):
        observation = np.ones(2) * (idx + 1)
        action = agent_with_mock_optimizer.act(observation)
        assert action < 2
        agent_with_mock_optimizer.update(
            {
                "action": action,
                "observation": observation,
                "reward": 1,
                "done": False,
            }
        )
    assert agent_with_mock_optimizer._optimizer.step.call_count == 0
    assert agent_with_mock_optimizer._replay_buffer.size() == 0
    assert agent_with_mock_optimizer._epsilon_schedule._value == pytest.approx(1.045)


def test_target_net_soft_update(agent_with_mock_optimizer):
    # Set the initial value of the parameters of the q-networks
    qnet_dict = agent_with_mock_optimizer._qnet.state_dict()
    target_net_dict = agent_with_mock_optimizer._target_qnet.state_dict()
    for key in qnet_dict:
        qnet_dict[key] = torch.ones_like(qnet_dict[key])
        target_net_dict[key] = torch.zeros_like(target_net_dict[key])
    agent_with_mock_optimizer._qnet.load_state_dict(qnet_dict)
    agent_with_mock_optimizer._target_qnet.load_state_dict(target_net_dict)

    # Run the network until its time to update the target network
    agent_with_mock_optimizer.train()
    for idx in range(5):
        # Assert that the target network hasn't changed
        check_target_network_value(agent_with_mock_optimizer._target_qnet, 0.0)

        observation = np.ones(2) * (idx + 1)
        action = agent_with_mock_optimizer.act(observation)
        assert action < 2
        agent_with_mock_optimizer.update(
            {
                "action": action,
                "observation": observation,
                "reward": 1,
                "done": False,
            }
        )

    # Assert that the target network was updated successfully
    check_target_network_value(agent_with_mock_optimizer._target_qnet, 0.25)


def test_target_net_hard_update(agent_with_mock_optimizer):
    # Set the initial value of the parameters of the q-networks
    agent_with_mock_optimizer._target_net_soft_update = False
    qnet_dict = agent_with_mock_optimizer._qnet.state_dict()
    target_net_dict = agent_with_mock_optimizer._target_qnet.state_dict()
    for key in qnet_dict:
        qnet_dict[key] = torch.ones_like(qnet_dict[key])
        target_net_dict[key] = torch.zeros_like(target_net_dict[key])
    agent_with_mock_optimizer._qnet.load_state_dict(qnet_dict)
    agent_with_mock_optimizer._target_qnet.load_state_dict(target_net_dict)

    # Run the network until its time to update the target network
    agent_with_mock_optimizer.train()
    observation = np.ones(2)
    for idx in range(5):
        # Assert that the target network hasn't changed
        check_target_network_value(agent_with_mock_optimizer._target_qnet, 0.0)
        observation = np.ones(2) * (idx + 1)
        action = agent_with_mock_optimizer.act(observation)
        assert action < 2
        agent_with_mock_optimizer.update(
            {
                "action": action,
                "observation": observation,
                "reward": 1,
                "done": False,
            }
        )

    # Assert that the target network was updated successfully
    check_target_network_value(agent_with_mock_optimizer._target_qnet, 1.0)


def test_save_load(agent_with_optimizer, tmpdir):
    agent_1 = agent_with_optimizer
    agent_2 = deepcopy(agent_with_optimizer)
    agent_1.train()

    # Run agent_1 so that it's internal state is different than agent_2
    for idx in range(10):
        observation = np.ones(2) * (idx + 1)
        action = agent_1.act(observation)
        assert action < 2
        agent_1.update(
            {
                "action": action,
                "observation": observation,
                "reward": 1,
                "done": False,
            }
        )

    # Make sure agent_1 and agent_2 have different internal states
    assert not check_dicts_equal(agent_1._qnet.state_dict(), agent_2._qnet.state_dict())
    assert not check_dicts_equal(
        agent_1._target_qnet.state_dict(), agent_2._target_qnet.state_dict()
    )
    assert not check_dicts_equal(
        agent_1._optimizer.state_dict(), agent_2._optimizer.state_dict()
    )
    assert agent_1._learn_schedule._steps != agent_2._learn_schedule._steps
    assert agent_1._epsilon_schedule._value != agent_2._epsilon_schedule._value
    assert (
        agent_1._target_net_update_schedule._steps
        != agent_2._target_net_update_schedule._steps
    )
    assert agent_1._rng.bit_generator.state != agent_2._rng.bit_generator.state

    # Save agent_1 and load it's state back into agent_2
    save_dir = tmpdir.mkdir("agent")
    agent_1.save(save_dir)
    agent_2.load(save_dir)

    # Make sure that agent_1 and agent_2 have the same internal state
    assert check_dicts_equal(agent_1._qnet.state_dict(), agent_2._qnet.state_dict())
    assert check_dicts_equal(
        agent_1._target_qnet.state_dict(), agent_2._target_qnet.state_dict()
    )
    assert check_dicts_equal(
        agent_1._optimizer.state_dict(), agent_2._optimizer.state_dict()
    )
    assert agent_1._learn_schedule._steps == agent_2._learn_schedule._steps
    assert agent_1._epsilon_schedule._value == agent_2._epsilon_schedule._value
    assert (
        agent_1._target_net_update_schedule._steps
        == agent_2._target_net_update_schedule._steps
    )
    assert agent_1._rng.bit_generator.state == agent_2._rng.bit_generator.state


def check_dicts_equal(dict_1, dict_2):
    """Used to recursively check if two dictionaries are equal."""
    if list(dict_1.keys()) != list(dict_2.keys()):
        return False
    for key in dict_1:
        value_1 = dict_1[key]
        value_2 = dict_2[key]
        if isinstance(dict_1[key], dict) and check_dicts_equal(value_1, value_2):
            continue
        elif isinstance(dict_1[key], torch.Tensor):
            value_1 = value_1.numpy()
            value_2 = value_2.numpy()
        value_2 = pytest.approx(value_2)
        if value_1 != value_2:
            return False
    return True


def check_target_network_value(network, value):
    """Checks if every parameter for the network is equal to value"""
    state_dict = network.state_dict()
    for key in state_dict:
        expected_value = torch.ones_like(state_dict[key]).numpy() * value
