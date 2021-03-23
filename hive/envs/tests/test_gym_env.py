import pytest
import os

import gym
import numpy as np
from hive.envs.gym_env import GymEnv

test_environments = ["CartPole-v0", "MountainCar-v0"]


@pytest.fixture()
def get_gym_env(env_name):
    env = gym.make(env_name)
    return env


@pytest.mark.parametrize("env_name", test_environments)
def test_env_spec(env_name, get_gym_env):
    hive_env = GymEnv(env_name)
    mock_env = get_gym_env
    assert hive_env.env_spec.obs_dim == mock_env.observation_space.shape
    assert hive_env.env_spec.act_dim == mock_env.action_space.n


@pytest.mark.parametrize("env_name", test_environments)
def test_reset_func(env_name):
    hive_env = GymEnv(env_name)
    hive_observation, hive_turn = hive_env.reset()

    assert isinstance(hive_observation, np.ndarray)
    assert isinstance(hive_turn, int)
    assert hive_turn == 0
    assert hive_observation.shape == hive_env.env_spec.obs_dim


@pytest.mark.parametrize("env_name", test_environments)
def test_step_func(env_name):
    hive_env = GymEnv(env_name)
    for action in range(hive_env.env_spec.act_dim):
        hive_env.reset()
        hive_observation, hive_reward, hive_done, hive_turn, hive_info = hive_env.step(action)

        assert isinstance(hive_observation, np.ndarray)
        assert isinstance(hive_reward, float)
        assert isinstance(hive_done, bool)
        assert isinstance(hive_info, dict)
        assert isinstance(hive_turn, int)
        assert hive_turn == 0
        assert hive_observation.shape == hive_env.env_spec.obs_dim
