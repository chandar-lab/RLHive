from copy import deepcopy
from functools import partial
from unittest.mock import Mock
import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.optim import Adam

from hive.agents import get_agent, DRQNAgent, DQNAgent
from hive.agents.qnets import MLPNetwork
from hive.agents.qnets.base import FunctionApproximator
from hive.envs import EnvSpec
from hive.replays import RecurrentReplayBuffer
from hive.utils import schedule

### RecurrentReplayBuffer, ConvRNNNetwork,

@pytest.fixture
def env_spec():
    return EnvSpec("test_env", gym.spaces.Box(0, 1, (2,)), gym.spaces.Discrete(2))

def xxxx_agent_with_mock_optimizer(env_spec):
    agent = DRQNAgent(
        observation_space=env_spec.observation_space,
        action_space=env_spec.action_space,
        representation_net=partial(MLPNetwork, hidden_units=5),
        optimizer_fn=Mock(),
        replay_buffer=partial(RecurrentReplayBuffer, capacity=10),
        target_net_update_fraction=0.25,
        target_net_soft_update=True,
        target_net_update_schedule=lambda: schedule.PeriodicSchedule(False, True, 5),
        epsilon_schedule=lambda: schedule.LinearSchedule(1.0, 0.1, 20),
        min_replay_history=2,
        device="cpu",
        batch_size=2,
        max_seq_len=1
    )
    return agent
