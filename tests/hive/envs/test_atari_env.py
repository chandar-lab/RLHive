import numpy as np
import pytest
from functools import partial 
from hive.envs import GymEnv
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from hive.envs.gym.gym_wrappers import PermuteImageWrapper

test_env_configs = [("ALE/Pong-v5", 4, 84), ("ALE/Asterix-v5", 1, 100)]


@pytest.mark.parametrize("env_name,frame_skip,screen_size", test_env_configs)
def test_reset_func(env_name, frame_skip, screen_size):
    hive_env = GymEnv(
        env_name, 
        repeat_action_probability=0.25,
        frameskip=1,
        env_wrappers=[
            partial(AtariPreprocessing, frame_skip=frame_skip, screen_size=screen_size, grayscale_newaxis=True),
            PermuteImageWrapper
        ]
    )
    hive_observation, hive_turn = hive_env.reset()

    assert isinstance(hive_observation, np.ndarray)
    assert isinstance(hive_turn, int)
    assert hive_turn == 0
    assert hive_observation.shape == hive_env.env_spec.observation_space[0].shape
    assert hive_observation.shape[0] == 1
    assert hive_observation.shape[1] == screen_size
    assert hive_observation.shape[2] == screen_size


@pytest.mark.parametrize("env_name,frame_skip,screen_size", test_env_configs)
def test_step_func(env_name, frame_skip, screen_size):
    hive_env = GymEnv(
        env_name, 
        repeat_action_probability=0.25,
        frameskip=1,
        env_wrappers=[
            partial(AtariPreprocessing, frame_skip=frame_skip, screen_size=screen_size, grayscale_newaxis=True),
            PermuteImageWrapper
        ]
    )
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
        assert hive_observation.shape[0] == 1
        assert hive_observation.shape[1] == screen_size
        assert hive_observation.shape[2] == screen_size

    init_observation, _ = hive_env.reset()
    for _ in range(50):
        hive_observation, _, _, _, _, _ = hive_env.step(
            hive_env.env_spec.action_space[0].sample()
        )
    assert (init_observation == hive_observation).all() == False
