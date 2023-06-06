from copy import deepcopy
from functools import partial
from unittest.mock import Mock
import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.optim import Adam

from hive.agents import DQNAgent, RainbowDQNAgent, get_agent
from hive.agents.qnets import MLPNetwork
from hive.agents.qnets.base import FunctionApproximator
from hive.envs import EnvSpec
from hive.replays import SimpleReplayBuffer
from hive.utils import schedule
from hive.utils.utils import Counter


@pytest.fixture
def env_spec():
    return EnvSpec("test_env", gym.spaces.Box(0, 1, (2,)), gym.spaces.Discrete(2))


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
    agent = RainbowDQNAgent(
        observation_space=env_spec.observation_space,
        action_space=env_spec.action_space,
        representation_net=partial(MLPNetwork, hidden_units=5, noisy=True),
        optimizer_fn=Mock(),
        replay_buffer=partial(SimpleReplayBuffer, capacity=10),
        target_net_update_fraction=0.25,
        target_net_soft_update=True,
        target_net_update_schedule=lambda: schedule.PeriodicSchedule(False, True, 5),
        epsilon_schedule=lambda: schedule.LinearSchedule(1.0, 0.1, 20),
        min_replay_history=2,
        device="cpu",
        batch_size=2,
        use_eps_greedy=True,
        double=True,
        dueling=True,
        noisy=True,
        distributional=True,
        atoms=51,
        v_min=0,
        v_max=200,
    )
    return agent


@pytest.fixture
def dxxx_agent_with_mock_optimizer(env_spec):
    agent = RainbowDQNAgent(
        observation_space=env_spec.observation_space,
        action_space=env_spec.action_space,
        representation_net=partial(MLPNetwork, hidden_units=5),
        optimizer_fn=Mock(),
        replay_buffer=partial(SimpleReplayBuffer, capacity=10),
        target_net_update_fraction=0.25,
        target_net_soft_update=True,
        target_net_update_schedule=lambda: schedule.PeriodicSchedule(False, True, 5),
        epsilon_schedule=lambda: schedule.LinearSchedule(1.0, 0.1, 20),
        min_replay_history=2,
        device="cpu",
        batch_size=2,
        use_eps_greedy=True,
        double=True,
        dueling=False,
        noisy=False,
        distributional=False,
    )
    return agent


@pytest.fixture
def xdxx_agent_with_mock_optimizer(env_spec):
    agent = RainbowDQNAgent(
        observation_space=env_spec.observation_space,
        action_space=env_spec.action_space,
        representation_net=partial(MLPNetwork, hidden_units=5),
        optimizer_fn=Mock(),
        replay_buffer=partial(SimpleReplayBuffer, capacity=10),
        target_net_update_fraction=0.25,
        target_net_soft_update=True,
        target_net_update_schedule=lambda: schedule.PeriodicSchedule(False, True, 5),
        epsilon_schedule=lambda: schedule.LinearSchedule(1.0, 0.1, 20),
        min_replay_history=2,
        device="cpu",
        use_eps_greedy=True,
        batch_size=2,
        double=False,
        dueling=True,
        noisy=False,
        distributional=False,
    )
    return agent


@pytest.fixture
def xxnx_agent_with_mock_optimizer(env_spec):
    agent = RainbowDQNAgent(
        observation_space=env_spec.observation_space,
        action_space=env_spec.action_space,
        representation_net=partial(MLPNetwork, hidden_units=5, noisy=True),
        optimizer_fn=Mock(),
        replay_buffer=partial(SimpleReplayBuffer, capacity=10),
        target_net_update_fraction=0.25,
        target_net_soft_update=True,
        target_net_update_schedule=lambda: schedule.PeriodicSchedule(False, True, 5),
        epsilon_schedule=lambda: schedule.LinearSchedule(1.0, 0.1, 20),
        min_replay_history=2,
        device="cpu",
        batch_size=2,
        use_eps_greedy=True,
        double=False,
        dueling=False,
        noisy=True,
        distributional=False,
    )
    return agent


@pytest.fixture
def xxxd_agent_with_mock_optimizer(env_spec):
    agent = RainbowDQNAgent(
        observation_space=env_spec.observation_space,
        action_space=env_spec.action_space,
        representation_net=partial(MLPNetwork, hidden_units=5),
        optimizer_fn=Mock(),
        replay_buffer=partial(SimpleReplayBuffer, capacity=10),
        target_net_update_fraction=0.25,
        target_net_soft_update=True,
        target_net_update_schedule=lambda: schedule.PeriodicSchedule(False, True, 5),
        epsilon_schedule=lambda: schedule.LinearSchedule(1.0, 0.1, 20),
        min_replay_history=2,
        device="cpu",
        batch_size=2,
        use_eps_greedy=True,
        double=False,
        dueling=False,
        noisy=False,
        distributional=True,
        atoms=51,
        v_min=0,
        v_max=200,
    )
    return agent


@pytest.fixture
def xxxx_agent_with_mock_optimizer(env_spec):
    agent = DQNAgent(
        observation_space=env_spec.observation_space,
        action_space=env_spec.action_space,
        representation_net=partial(MLPNetwork, hidden_units=5),
        optimizer_fn=Mock(),
        replay_buffer=partial(SimpleReplayBuffer, capacity=10),
        target_net_update_fraction=0.25,
        target_net_soft_update=True,
        target_net_update_schedule=lambda: schedule.PeriodicSchedule(False, True, 5),
        epsilon_schedule=lambda: schedule.LinearSchedule(1.0, 0.1, 20),
        min_replay_history=2,
        device="cpu",
        batch_size=2,
    )
    return agent


@pytest.fixture
def xxxx_rainbow_agent_with_mock_optimizer(env_spec):
    agent = RainbowDQNAgent(
        observation_space=env_spec.observation_space,
        action_space=env_spec.action_space,
        representation_net=partial(MLPNetwork, hidden_units=5),
        optimizer_fn=Mock(),
        replay_buffer=partial(SimpleReplayBuffer, capacity=10),
        target_net_update_fraction=0.25,
        target_net_soft_update=True,
        target_net_update_schedule=lambda: schedule.PeriodicSchedule(False, True, 5),
        epsilon_schedule=lambda: schedule.LinearSchedule(1.0, 0.1, 20),
        min_replay_history=2,
        device="cpu",
        batch_size=2,
        use_eps_greedy=False,
        double=False,
        dueling=False,
        noisy=False,
        distributional=False,
    )
    return agent


@pytest.fixture
def agent_with_optimizer(env_spec):
    agent = DQNAgent(
        observation_space=env_spec.observation_space,
        action_space=env_spec.action_space,
        representation_net=partial(MLPNetwork, hidden_units=5),
        optimizer_fn=Adam,
        replay_buffer=partial(SimpleReplayBuffer, capacity=10),
        target_net_update_fraction=0.25,
        target_net_soft_update=True,
        target_net_update_schedule=lambda: schedule.PeriodicSchedule(False, True, 5),
        epsilon_schedule=lambda: schedule.LinearSchedule(1.0, 0.1, 20),
        min_replay_history=2,
        device="cpu",
        batch_size=2,
    )
    return agent


def test_create_agent_with_configs(env_spec):
    agent_config = {
        "name": "DQNAgent",
        "kwargs": {
            "observation_space": env_spec.observation_space,
            "action_space": env_spec.action_space,
            "representation_net": {
                "name": "MLPNetwork",
                "kwargs": {"hidden_units": 5},
            },
            "optimizer_fn": {"name": "Adam", "kwargs": {"lr": 0.01}},
            "replay_buffer": {
                "name": "SimpleReplayBuffer",
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
            "min_replay_history": 2,
            "device": "cpu",
        },
    }
    agent, _ = get_agent(agent_config)
    agent = agent()
    agent_traj_state = None
    action, agent_traj_state = agent.act(np.zeros(2), agent_traj_state, Counter())
    assert action < 2


def test_train_step(agent_with_mock_optimizer):
    agent_with_mock_optimizer.train()
    agent_traj_state = None
    episode_step = Counter()
    for idx in range(8):
        observation = np.ones(2) * (idx + 1)
        next_observation = np.ones(2) * (idx + 2)
        action, agent_traj_state = agent_with_mock_optimizer.act(
            observation, agent_traj_state, episode_step
        )
        assert action < 2
        agent_traj_state = agent_with_mock_optimizer.update(
            {
                "action": action,
                "observation": observation,
                "next_observation": next_observation,
                "reward": 1,
                "terminated": False,
                "truncated": False,
                "source": 0,
            },
            agent_traj_state,
            episode_step,
        )
        episode_step.increment()
    assert agent_with_mock_optimizer._optimizer.step.call_count == 6
    assert agent_with_mock_optimizer._replay_buffer.size() == 8
    assert episode_step == 8
    assert agent_with_mock_optimizer._epsilon_schedule(episode_step) == pytest.approx(
        0.64
    )
    assert episode_step == 8


def test_eval_step(agent_with_mock_optimizer):
    agent_with_mock_optimizer.eval()
    agent_traj_state = None
    episode_step = Counter()
    for idx in range(8):
        observation = np.ones(2) * (idx + 1)
        next_observation = np.ones(2) * (idx + 2)
        action, agent_traj_state = agent_with_mock_optimizer.act(
            observation, agent_traj_state, episode_step
        )
        assert action < 2
        agent_traj_state = agent_with_mock_optimizer.update(
            {
                "action": action,
                "observation": observation,
                "next_observation": next_observation,
                "reward": 1,
                "terminated": False,
                "truncated": False,
                "source": 0,
            },
            agent_traj_state,
            episode_step,
        )
    assert agent_with_mock_optimizer._optimizer.step.call_count == 0
    assert agent_with_mock_optimizer._replay_buffer.size() == 0
    assert agent_with_mock_optimizer._epsilon_schedule(episode_step) == pytest.approx(
        1.0
    )
    assert episode_step == 0


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
    agent_traj_state = None

    episode_step = Counter()
    for idx in range(5):
        # Assert that the target network hasn't changed
        check_target_network_value(agent_with_mock_optimizer._target_qnet, 0.0)

        observation = np.ones(2) * (idx + 1)
        next_observation = np.ones(2) * (idx + 2)
        action, agent_traj_state = agent_with_mock_optimizer.act(
            observation, agent_traj_state, episode_step
        )
        assert action < 2
        agent_traj_state = agent_with_mock_optimizer.update(
            {
                "action": action,
                "observation": observation,
                "next_observation": next_observation,
                "reward": 1,
                "terminated": False,
                "truncated": False,
                "source": 0,
            },
            agent_traj_state,
            episode_step,
        )
        episode_step.increment()

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
    agent_traj_state = None
    episode_step = Counter()
    for idx in range(5):
        # Assert that the target network hasn't changed
        check_target_network_value(agent_with_mock_optimizer._target_qnet, 0.0)
        observation = np.ones(2) * (idx + 1)
        next_observation = np.ones(2) * (idx + 2)
        action, agent_traj_state = agent_with_mock_optimizer.act(
            observation, agent_traj_state, episode_step
        )
        assert action < 2
        agent_traj_state = agent_with_mock_optimizer.update(
            {
                "action": action,
                "observation": observation,
                "next_observation": next_observation,
                "reward": 1,
                "terminated": False,
                "truncated": False,
                "source": 0,
            },
            agent_traj_state,
            episode_step,
        )
        episode_step.increment()

    # Assert that the target network was updated successfully
    check_target_network_value(agent_with_mock_optimizer._target_qnet, 1.0)


def test_save_load(agent_with_optimizer, tmpdir):
    agent_1 = agent_with_optimizer
    agent_2 = deepcopy(agent_with_optimizer)
    agent_1.train()
    agent_1_traj_state = None
    agent_2_traj_state = None
    episode_1_step = Counter()
    episode_2_step = Counter()
    # Run agent_1 so that it's internal state is different than agent_2
    for idx in range(10):
        observation = np.ones(2) * (idx + 1)
        next_observation = np.ones(2) * (idx + 2)
        action, agent_1_traj_state = agent_1.act(
            observation, agent_1_traj_state, episode_1_step
        )
        assert action < 2
        agent_1_traj_state = agent_1.update(
            {
                "action": action,
                "observation": observation,
                "next_observation": next_observation,
                "reward": 1,
                "terminated": False,
                "truncated": False,
                "source": 0,
            },
            agent_1_traj_state,
            episode_1_step,
        )
        episode_1_step.increment()

    # Make sure agent_1 and agent_2 have different internal states
    assert not check_dicts_equal(agent_1._qnet.state_dict(), agent_2._qnet.state_dict())
    assert not check_dicts_equal(
        agent_1._target_qnet.state_dict(), agent_2._target_qnet.state_dict()
    )
    assert not check_dicts_equal(
        agent_1._optimizer.state_dict(), agent_2._optimizer.state_dict()
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
