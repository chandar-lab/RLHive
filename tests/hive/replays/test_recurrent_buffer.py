import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from hive.replays.recurrent_replay import RecurrentReplayBuffer


OBS_SHAPE = (2, 2)
CAPACITY = 60
MAX_SEQ_LEN = 10
N_STEP_HORIZON = 2
GAMMA = 1


@pytest.fixture()
def rec_buffer():
    return RecurrentReplayBuffer(
        capacity=CAPACITY,
        max_seq_len=MAX_SEQ_LEN,
        observation_shape=OBS_SHAPE,
        observation_dtype=np.float32,
        extra_storage_types={"priority": (np.int8, ())},
    )


@pytest.fixture(
    params=[
        pytest.lazy_fixture("rec_buffer"),
    ]
)
def buffer(request):
    return request.param


### truncated and terminated instead of done???
@pytest.fixture()
def full_buffer(buffer):
    done_time = 15
    for i in range(33):  # until the buffer is full instead of CAPACITY
        buffer.add(
            observation=np.ones(OBS_SHAPE) * i,
            action=(i % done_time) + 1,
            reward=(i % done_time) + 1,
            done=((i + 1) % done_time) == 0,
            priority=(i % 10) + 1,
        )
    return buffer


@pytest.fixture()
def full_n_step_buffer():
    n_step_buffer = RecurrentReplayBuffer(
        capacity=CAPACITY,
        max_seq_len=MAX_SEQ_LEN,
        observation_shape=OBS_SHAPE,
        observation_dtype=np.float32,
        n_step=N_STEP_HORIZON,
        gamma=GAMMA,
    )
    done_time = 15
    for i in range(33):  # until the buffer is full instead of CAPACITY
        n_step_buffer.add(
            observation=np.ones(OBS_SHAPE) * i,
            action=(i % done_time) + 1,
            reward=(i % done_time) + 1,
            done=((i + 1) % done_time) == 0,
            priority=(i % 10) + 1,
        )
    return n_step_buffer


@pytest.mark.parametrize("constructor", [RecurrentReplayBuffer])
@pytest.mark.parametrize("observation_shape", [(), (2,), (3, 4)])
@pytest.mark.parametrize("observation_dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("action_shape", [(), (5,)])
@pytest.mark.parametrize("action_dtype", [np.int8, np.float32])
@pytest.mark.parametrize("reward_shape", [(), (6,)])
@pytest.mark.parametrize("reward_dtype", [np.int8, np.float32])
@pytest.mark.parametrize("extra_storage_types", [None, {"foo": (np.float32, (7,))}])
def test_constructor(
    constructor,
    observation_shape,
    observation_dtype,
    action_shape,
    action_dtype,
    reward_shape,
    reward_dtype,
    extra_storage_types,
):
    buffer = constructor(
        capacity=10,
        max_seq_len=MAX_SEQ_LEN,
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        action_shape=action_shape,
        action_dtype=action_dtype,
        reward_shape=reward_shape,
        reward_dtype=reward_dtype,
        extra_storage_types=extra_storage_types,
    )
    assert buffer.size() == 0
    assert buffer._max_seq_len == MAX_SEQ_LEN
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
    ## add to the buffer until the buffer is full
    done_time = 15
    for i in range(33):  # until the buffer is full instead of CAPACITY
        buffer.add(
            observation=np.ones(OBS_SHAPE) * i,
            action=i,
            reward=i % 10,
            done=((i + 1) % done_time) == 0,
            priority=(i % 10) + 1,
        )

        assert buffer.size() == i + ((i) // done_time) * (MAX_SEQ_LEN - 1)
        assert (
            buffer._cursor
            == ((i + MAX_SEQ_LEN) + (((i) // done_time) * (MAX_SEQ_LEN - 1))) % CAPACITY
        )

    ## when the buffer is full
    more_steps = 40
    for i in range(more_steps):
        buffer.add(
            observation=np.ones(OBS_SHAPE) * i,
            action=i,
            reward=i % 10,
            done=((i + 1) % done_time) == 0,
            priority=(i % 10) + 1,
        )
        assert buffer.size() == CAPACITY - MAX_SEQ_LEN
        assert buffer._cursor == (
            ((i + 1) + (((i) // done_time) * (MAX_SEQ_LEN - 1))) % CAPACITY
        )

    assert (
        buffer._num_added
        == (more_steps + (((i) // done_time) * (MAX_SEQ_LEN - 1))) + CAPACITY
    )


def test_sample_shape(full_buffer):
    # sample transitions from buffer
    batch_size = CAPACITY - 1
    batch = full_buffer.sample(batch_size)
    # check if the shape of batch is correct
    assert batch["indices"].shape == (batch_size,)
    assert batch["observation"].shape == (batch_size, 10) + OBS_SHAPE
    assert batch["action"].shape == (batch_size, 10)
    assert batch["done"].shape == (batch_size, 10)
    assert batch["reward"].shape == (batch_size, 10)
    assert batch["trajectory_lengths"].shape == (batch_size,)
    assert batch["next_observation"].shape == (batch_size, 10) + OBS_SHAPE


def test_sample(full_buffer):
    # sample transitions from buffer
    batch_size = 50
    batch = full_buffer.sample(batch_size)
    for b in range(batch_size):
        t = 0

        while batch["action"][b, t] == 0:
            t += 1

        while t < MAX_SEQ_LEN and batch["action"][b, t] > 0:
            if t > 0:
                assert batch["action"][b, t] - batch["action"][b, t - 1] == 1
                assert batch["reward"][b, t] - batch["reward"][b, t - 1] == 1
            t += 1


def test_sample_n_step(full_n_step_buffer):
    # sample transitions from buffer
    batch_size = 50
    batch = full_n_step_buffer.sample(batch_size)
    for b in range(batch_size):
        t = 0

        while batch["action"][b, t] == 0:
            t += 1

        while t < MAX_SEQ_LEN - 1 and batch["action"][b, t] > 0:
            if t > 0:
                assert batch["action"][b, t] - batch["action"][b, t - 1] == 1
                if t == full_n_step_buffer.size() - 1 or batch["reward"][b, t + 1] == 0:
                    assert (
                        batch["reward"][b, t] - batch["reward"][b, t - 1]
                        == 1 - batch["reward"][b, t] * GAMMA
                    )
                else:
                    assert (
                        batch["reward"][b, t] - batch["reward"][b, t - 1] == 1 + GAMMA
                    )
            t += 1
