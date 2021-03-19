# Create dqn with objects
# Create dqn with configs
# Run train step, confirm change in state, confirm stuff added to replay
# Run eval step, confirm no change in state, replay
# Test syncing
# Test saving/loading
import pytest
from unittest.mock import Mock
import torch
from hive.envs import EnvSpec
from hive.agents import DQNAgent
from hive.agents.qnets import SimpleMLP
from torch.optim import Adam
from hive.replays import CircularReplayBuffer
from hive.utils import schedule
import numpy as np


@pytest.fixture
def env_spec():
    return EnvSpec("test_env", (2,), 2)


@pytest.fixture
def agent_with_mock_optimizer(env_spec):
    agent = DQNAgent(
        qnet=SimpleMLP(env_spec, hidden_units=5, num_hidden_layers=1),
        env_spec=env_spec,
        optimizer_fn=Mock(),
        replay_buffer=CircularReplayBuffer(size=10),
        target_net_update_fraction=0.25,
        target_net_soft_update=True,
        target_net_update_schedule=schedule.PeriodicSchedule(False, True, 5),
        epsilon_schedule=schedule.LinearSchedule(1.0, 0.1, 20),
        learn_schedule=schedule.SwitchSchedule(False, True, 2),
        device="cpu",
        batch_size=2,
    )
    return agent


def test_create_agent_with_objects(env_spec):
    agent = DQNAgent(
        qnet=SimpleMLP(env_spec, hidden_units=5, num_hidden_layers=1),
        env_spec=env_spec,
        optimizer_fn=Adam,
        replay_buffer=CircularReplayBuffer(size=10),
        target_net_update_schedule=schedule.PeriodicSchedule(False, True, 5),
        epsilon_schedule=schedule.LinearSchedule(1.0, 0.1, 20),
        learn_schedule=schedule.SwitchSchedule(False, True, 2),
        device="cpu",
    )
    action = agent.act(np.zeros(2))
    assert action < 2


def test_create_agent_with_configs(env_spec):
    agent = DQNAgent(
        qnet={
            "name": "SimpleMLP",
            "kwargs": {"hidden_units": 5, "num_hidden_layers": 1},
        },
        env_spec=env_spec,
        optimizer_fn={"name": "Adam", "kwargs": {"lr": 0.01}},
        replay_buffer={"name": "CircularReplayBuffer", "kwargs": {"size": 10}},
        target_net_update_schedule={
            "name": "PeriodicSchedule",
            "kwargs": {"off_value": False, "on_value": True, "period": 5},
        },
        epsilon_schedule={
            "name": "LinearSchedule",
            "kwargs": {"init_value": 1.0, "end_value": 0.1, "steps": 20},
        },
        learn_schedule={
            "name": "SwitchSchedule",
            "kwargs": {"off_value": False, "on_value": True, "steps": 2},
        },
        device="cpu",
    )
    action = agent.act(np.zeros(2))
    assert action < 2


def test_train_step(agent_with_mock_optimizer):
    agent_with_mock_optimizer.train()
    observation = np.ones(2)
    for idx in range(8):
        action = agent_with_mock_optimizer.act(observation)
        assert action < 2
        next_observation = np.ones(2) * (idx + 1)
        agent_with_mock_optimizer.update(
            {
                "action": action,
                "observation": observation,
                "next_observation": next_observation,
                "reward": 1,
                "done": False,
            }
        )
        observation = next_observation
    assert agent_with_mock_optimizer._optimizer.step.call_count == 7
    assert agent_with_mock_optimizer._replay_buffer.size() == 8
    assert agent_with_mock_optimizer._epsilon_schedule._value == pytest.approx(0.775)


def test_eval_step(agent_with_mock_optimizer):
    agent_with_mock_optimizer.eval()
    observation = np.ones(2)
    for idx in range(8):
        action = agent_with_mock_optimizer.act(observation)
        assert action < 2
        next_observation = np.ones(2) * (idx + 1)
        agent_with_mock_optimizer.update(
            {
                "action": action,
                "observation": observation,
                "next_observation": next_observation,
                "reward": 1,
                "done": False,
            }
        )
        observation = next_observation
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
    observation = np.ones(2)
    for idx in range(5):
        # Assert that the target network hasn't changed
        check_target_network_value(agent_with_mock_optimizer._target_qnet, 0.0)

        action = agent_with_mock_optimizer.act(observation)
        assert action < 2
        next_observation = np.ones(2) * (idx + 1)
        agent_with_mock_optimizer.update(
            {
                "action": action,
                "observation": observation,
                "next_observation": next_observation,
                "reward": 1,
                "done": False,
            }
        )
        observation = next_observation

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
        action = agent_with_mock_optimizer.act(observation)
        assert action < 2
        next_observation = np.ones(2) * (idx + 1)
        agent_with_mock_optimizer.update(
            {
                "action": action,
                "observation": observation,
                "next_observation": next_observation,
                "reward": 1,
                "done": False,
            }
        )
        observation = next_observation

    # Assert that the target network was updated successfully
    check_target_network_value(agent_with_mock_optimizer._target_qnet, 1.0)


def check_target_network_value(network, value):
    """Checks if every parameter for the network is equal to value"""
    state_dict = network.state_dict()
    for key in state_dict:
        expected_value = torch.ones_like(state_dict[key]).numpy() * value
        assert state_dict[key].numpy() == pytest.approx(expected_value)
