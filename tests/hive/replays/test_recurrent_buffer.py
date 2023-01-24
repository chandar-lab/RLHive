
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from hive.replays.recurrent_replay import RecurrentReplayBuffer

### size, seq_length,
### add --> check shape,

OBS_SHAPE = (4, 4)
CAPACITY = 60
MAX_SEQ_LEN = 10
N_STEP_HORIZON = 3
GAMMA = 0.99

@pytest.fixture()
def rec_buffer():
    return RecurrentReplayBuffer(
        capacity=CAPACITY,
        max_seq_len = MAX_SEQ_LEN,
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
    for i in range(CAPACITY + 20):
        buffer.add(
            observation=np.ones(OBS_SHAPE) * i,
            action=i,
            reward=i % 10,
            done=((i + 1) % 15) == 0,
            priority=(i % 10) + 1,
        )
    return buffer



@pytest.fixture()
def full_n_step_buffer():
    n_step_buffer = RecurrentReplayBuffer(
        capacity=CAPACITY,
        observation_shape=OBS_SHAPE,
        observation_dtype=np.float32,
        n_step=N_STEP_HORIZON,
        gamma=GAMMA,
    )
    for i in range(CAPACITY + 20):
        n_step_buffer.add(
            observation=np.ones(OBS_SHAPE) * i,
            action=i,
            reward=i % 10,
            done=((i + 1) % 15) == 0,
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
    assert buffer._max_seq_len == 10
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
            priority=(i % 10) + 1,
        )
        if i < MAX_SEQ_LEN:
            assert buffer.size() == i
            assert buffer._cursor == ((i + 1) % CAPACITY)
        else:
            assert buffer.size() == i - MAX_SEQ_LEN
            assert buffer._cursor == ((i + 1) % CAPACITY)

    for i in range(20):
        buffer.add(
            observation=np.ones(OBS_SHAPE) * i,
            action=i,
            reward=i % 10,
            done=((i + 1) % 15) == 0,
            priority=(i % 10) + 1,
        )
        assert buffer.size() == CAPACITY - MAX_SEQ_LEN- 1
        assert buffer._cursor == ((i + 1) % CAPACITY)
    assert buffer._num_added == CAPACITY + 20







