import pytest

import numpy as np
from hive.envs.atari import AtariEnv

test_env_configs = [("Pong", 4, 84), ("Breakout", 1, 100)]


@pytest.mark.parametrize("env_name,frame_skip,screen_size", test_env_configs)
def test_reset_func(env_name, frame_skip, screen_size):
    hive_env = AtariEnv(env_name, frame_skip, screen_size)
    hive_observation, hive_turn = hive_env.reset()

    assert isinstance(hive_observation, np.ndarray)
    assert isinstance(hive_turn, int)
    assert hive_turn == 0
    assert hive_observation.shape == hive_env.env_spec.obs_dim[0]
    assert hive_observation.shape[0] == 1
    assert hive_observation.shape[1] == screen_size
    assert hive_observation.shape[2] == screen_size


@pytest.mark.parametrize("env_name,frame_skip,screen_size", test_env_configs)
def test_step_func(env_name, frame_skip, screen_size):
    hive_env = AtariEnv(env_name, frame_skip, screen_size)
    for action in range(hive_env.env_spec.act_dim[0]):
        hive_env.reset()
        hive_observation, hive_reward, hive_done, hive_turn, hive_info = hive_env.step(
            action
        )

        assert isinstance(hive_observation, np.ndarray)
        assert isinstance(hive_reward, float)
        assert isinstance(hive_done, bool)
        assert isinstance(hive_info, dict)
        assert isinstance(hive_turn, int)
        assert hive_turn == 0
        assert hive_observation.shape == hive_env.env_spec.obs_dim[0]
        assert hive_observation.shape[0] == 1
        assert hive_observation.shape[1] == screen_size
        assert hive_observation.shape[2] == screen_size

    init_observation, _ = hive_env.reset()
    for i in range(50):
        hive_observation, _, _, _, _ = hive_env.step(np.random.randint(hive_env.env_spec.act_dim[0]))
    assert (init_observation == hive_observation).all() == False
