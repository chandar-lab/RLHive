import os

import gymnasium as gym
import numpy as np
import pytest

from hive.envs.gym_env import GymEnv

test_environments = ["CartPole-v0", "MountainCar-v0"]


@pytest.fixture()
def gym_env(env_name):
    env = gym.make(env_name)
    return env


@pytest.mark.parametrize("env_name", test_environments)
def test_env_spec(env_name, gym_env):
    hive_env = GymEnv(env_name)
    assert hive_env.env_spec.observation_space[0] == gym_env.observation_space
    assert hive_env.env_spec.action_space[0] == gym_env.action_space


@pytest.mark.parametrize("env_name", test_environments)
def test_reset_func(env_name):
    hive_env = GymEnv(env_name)
    hive_observation, hive_turn = hive_env.reset()

    assert isinstance(hive_observation, np.ndarray)
    assert isinstance(hive_turn, int)
    assert hive_turn == 0
    assert hive_observation.shape == hive_env.env_spec.observation_space[0].shape


@pytest.mark.parametrize("env_name", test_environments)
def test_step_func(env_name):
    hive_env = GymEnv(env_name)
    for action in range(hive_env.env_spec.action_space[0].n):
        hive_env.reset()
        (
            hive_observation,
            hive_reward,
            hive_terminated,
            hive_truncated,
            hive_turn,
            hive_info,
        ) = hive_env.step(action)

        assert isinstance(hive_observation, np.ndarray)
        assert isinstance(hive_reward, float)
        assert isinstance(hive_terminated, bool)
        assert isinstance(hive_truncated, bool)
        assert isinstance(hive_info, dict)
        assert isinstance(hive_turn, int)
        assert hive_turn == 0
        assert hive_observation.shape == hive_env.env_spec.observation_space[0].shape
