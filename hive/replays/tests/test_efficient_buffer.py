import pytest
import os

import numpy as np
from hive.replays import EfficientCircularBuffer

OBS_SHAPE = (4, 4)
CAPACITY = 60
STACK_SIZE = 2


@pytest.fixture()
def buffer():
    return EfficientCircularBuffer(
        capacity=CAPACITY,
        observation_shape=OBS_SHAPE,
        observation_dtype=np.float32,
        extra_storage_types={"foo": (np.int8, ())},
    )


@pytest.fixture()
def full_buffer(buffer):
    for i in range(CAPACITY + 20):
        buffer.add(
            observation=np.ones(OBS_SHAPE) * i,
            action=i,
            reward=i % 10,
            done=((i + 1) % 15) == 0,
            foo=i % 5,
        )
    return buffer


@pytest.fixture()
def stacked_buffer():
    return EfficientCircularBuffer(
        capacity=CAPACITY,
        stack_size=STACK_SIZE,
        observation_shape=OBS_SHAPE,
        observation_dtype=np.float32,
    )


@pytest.fixture()
def full_stacked_buffer(stacked_buffer):
    for i in range(CAPACITY + 20):
        stacked_buffer.add(
            observation=np.ones(OBS_SHAPE) * i,
            action=i,
            reward=i % 10,
            done=((i + 1) % 15) == 0,
        )
    return stacked_buffer


@pytest.mark.parametrize("observation_shape", [(), (2,), (3, 4)])
@pytest.mark.parametrize("observation_dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("action_shape", [(), (5,)])
@pytest.mark.parametrize("action_dtype", [np.int8, np.float32])
@pytest.mark.parametrize("reward_shape", [(), (6,)])
@pytest.mark.parametrize("reward_dtype", [np.int8, np.float32])
@pytest.mark.parametrize("extra_storage_types", [None, {"foo": (np.float32, (7,))}])
def test_constructor(
    observation_shape,
    observation_dtype,
    action_shape,
    action_dtype,
    reward_shape,
    reward_dtype,
    extra_storage_types,
):
    buffer = EfficientCircularBuffer(
        capacity=10,
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        action_shape=action_shape,
        action_dtype=action_dtype,
        reward_shape=reward_shape,
        reward_dtype=reward_dtype,
        extra_storage_types=extra_storage_types,
    )
    assert buffer.size() == 0
    assert buffer._storage["observation"].shape == (10,) + observation_shape
    assert buffer._storage["observation"].dtype == observation_dtype
    assert buffer._storage["action"].shape == (10,) + action_shape
    assert buffer._storage["action"].dtype == action_dtype
    assert buffer._storage["reward"].shape == (10,) + reward_shape
    assert buffer._storage["reward"].dtype == reward_dtype
    if extra_storage_types is not None:
        for key in extra_storage_types:
            assert buffer._storage[key].shape == (10,) + extra_storage_types[key][1]
            assert buffer._storage[key].dtype == extra_storage_types[key][0]


def test_add(buffer):
    assert buffer.size() == 0
    for i in range(CAPACITY):
        buffer.add(
            observation=np.ones(OBS_SHAPE) * i,
            action=i,
            reward=i % 10,
            done=((i + 1) % 15) == 0,
            foo=i % 5,
        )
        assert buffer.size() == i
        assert buffer._cursor == ((i + 1) % CAPACITY)

    for i in range(20):
        buffer.add(
            observation=np.ones(OBS_SHAPE) * i,
            action=i,
            reward=i % 10,
            done=((i + 1) % 15) == 0,
            foo=i % 5,
        )
        assert buffer.size() == CAPACITY - 1
        assert buffer._cursor == ((i + 1) % CAPACITY)
    assert buffer._num_added == CAPACITY + 20


def test_sample(full_buffer):
    batch = full_buffer.sample(CAPACITY - 1)
    for i in range(CAPACITY - 1):
        timestep = batch["action"][i]
        assert batch["reward"][i] == timestep % 10
        assert batch["foo"][i] == timestep % 5
        assert batch["done"][i] == (((timestep + 1) % 15) == 0)
        assert batch["observation"][i] == pytest.approx(np.ones(OBS_SHAPE) * timestep)
        if not batch["done"][i]:
            assert batch["observation"][i] + 1 == pytest.approx(
                batch["next_observation"][i]
            )


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize("stack_size, num_added", [(1, 1), (2, 1), (2, 2)])
def test_sample_few_transitions(stack_size, num_added):
    buffer = EfficientCircularBuffer(
        capacity=CAPACITY,
        stack_size=STACK_SIZE,
        observation_shape=OBS_SHAPE,
        observation_dtype=np.float32,
    )
    for i in range(num_added):
        buffer.add(np.ones(OBS_SHAPE) * i, action=i, reward=i % 10, done=(i + 1) % 15)
    buffer.sample(1)


def test_save_load(full_buffer, tmpdir):
    save_dir = tmpdir.mkdir("replay")
    full_buffer.save(save_dir)
    buffer = EfficientCircularBuffer(
        capacity=CAPACITY,
        observation_shape=OBS_SHAPE,
        observation_dtype=np.float32,
        extra_storage_types={"foo": (np.int8, ())},
    )
    buffer.load(save_dir)
    for key in full_buffer._storage:
        assert full_buffer._storage[key] == pytest.approx(buffer._storage[key])

    assert buffer._episode_start == full_buffer._episode_start
    assert buffer._cursor == full_buffer._cursor
    assert buffer._num_added == full_buffer._num_added
    assert buffer._rng.bit_generator.state == full_buffer._rng.bit_generator.state

def test_stacked_buffer_add(stacked_buffer):
    assert stacked_buffer.size() == 0

    for i in range(CAPACITY):
        stacked_buffer.add(
            observation=np.ones(OBS_SHAPE) * i,
            action=i,
            reward=i % 10,
            done=((i + 1) % 15) == 0,
        )
        assert (
            stacked_buffer.size()
            == min(max(0, i + 2 + (i // 15)), CAPACITY) - STACK_SIZE
        )

    for i in range(20):
        stacked_buffer.add(
            observation=np.ones(OBS_SHAPE) * i,
            action=i,
            reward=i % 10,
            done=((i + 1) % 15) == 0,
        )
        assert stacked_buffer.size() == CAPACITY - STACK_SIZE
    assert stacked_buffer._num_added == CAPACITY + 20 + 1 + (CAPACITY + 20) // 15


def test_stacked_buffer_sample(full_stacked_buffer):
    assert full_stacked_buffer.size() == CAPACITY - STACK_SIZE
    batch_size = 10
    batch = full_stacked_buffer.sample(batch_size)
    for i in range(batch_size):
        timestep = batch["action"][i]
        assert batch["observation"][i].shape == ((STACK_SIZE,) + OBS_SHAPE)
        assert batch["reward"][i] == timestep % 10
        assert batch["done"][i] == (((timestep + 1) % 15) == 0)
        if not batch["done"][i]:
            assert batch["observation"][i, -1] == pytest.approx(
                batch["next_observation"][i, 0]
            )
            assert batch["observation"][i, -1] + 1 == pytest.approx(
                batch["next_observation"][i, 1]
            )
