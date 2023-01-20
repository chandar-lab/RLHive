
import numpy as np
import pytest

from hive.replays.recurrent_replay import RecurrentReplayBuffer

### size, seq_length,
### add --> check shape,

OBS_SHAPE = (4, 4)
CAPACITY = 60
MAX_SEQ_LEN = 20
N_STEP_HORIZON = 3
GAMMA = 0.99

@pytest.fixture()
def rec_buffer():
    return RecurrentReplayBuffer(
        capacity=CAPACITY,
        max_seq_len = MAX_SEQ_LEN,
        observation_shape=OBS_SHAPE,
        observation_dtype=np.float32,
        # extra_storage_types={"priority": (np.int8, ())},

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
    n_step_buffer = CircularReplayBuffer(
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

