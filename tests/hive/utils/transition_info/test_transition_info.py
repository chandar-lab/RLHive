import os
import sys
from argparse import Namespace
from unittest.mock import patch

import pytest

import hive
from hive.runners.utils import load_config
from hive.runners.utils import TransitionInfo
from hive.agents.dqn import DQNAgent
from hive.agents.qnets.mlp import MLPNetwork
import gym
import numpy as np

@pytest.fixture()
def args():
    return Namespace(
        config="tests/hive/utils/transition_info/test_transition_info_config.yml",
        agent_config=None,
    )

@pytest.fixture()
def transition_info(args, tmpdir):
    config = load_config(
        args.config,
        agent_config=args.agent_config,
    )
    config["save_dir"] = os.path.join(tmpdir, config["save_dir"])
    env = gym.make('CartPole-v0')
    agent_config = config['agent']
    
    agent0 = DQNAgent(observation_space = env.observation_space,
                     action_space = env.action_space,
                     representation_net = MLPNetwork,
                     id = 0
                    )
    agent1 = DQNAgent(observation_space = env.observation_space,
                     action_space = env.action_space,
                     representation_net = MLPNetwork,
                     id = 1
                    )

    agents = [agent0, agent1]
    stack_size = 5
    t_info = TransitionInfo(agents, stack_size)
    return t_info, agents, config

def test_start_agent(transition_info):
    t_info, agents, config = transition_info
    t_info.start_agent(agents[0])
    assert t_info._started[agents[0].id] == True

def test_is_started(transition_info):
    t_info, agents, config = transition_info
    t_info.start_agent(agents[0])
    assert t_info.is_started(agents[0]) == True
    assert t_info.is_started(agents[1]) == False

def test_update_reward(transition_info):
    t_info, agents, config = transition_info
    t_info.start_agent(agents[0])
    t_info.update_reward(agents[0], 1.)
    assert t_info._transitions[t_info._agent_ids[0]]["reward"] == 1.

def test_update_all_rewards(transition_info):
    t_info, agents, config = transition_info
    rewards = [1., 2.]
    t_info.update_all_rewards(rewards)
    assert t_info._transitions[t_info._agent_ids[0]]["reward"] == 1.
    assert t_info._transitions[t_info._agent_ids[1]]["reward"] == 2.

def test_get_info(transition_info):
    t_info, agents, config = transition_info
    info = t_info.get_info(agents[0], terminated = True, truncated = True)
    assert info == {'reward':0, 'truncated':True, 'terminated' : True}
    assert t_info._transitions['0'] == {"reward": 0.0}

def test_record_info(transition_info):
    t_info, agents, config = transition_info
    info = {'observation': 2, 'reward':1}
    t_info.record_info(agents[0], info)
    assert t_info._transitions['0'] == {'observation': 2, 'reward':1}
    assert t_info._previous_observations['0'][-1] == 2

def test_get_stacked_state(transition_info):
    t_info, agents, config = transition_info
    observation = 2
    t_info._previous_observations[agents[0].id].append(3)
    stacked_observation = t_info.get_stacked_state(agents[0], observation)
    assert list(stacked_observation) == [3,0,2]
