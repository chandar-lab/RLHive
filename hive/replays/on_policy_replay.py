import numpy as np

from hive.replays.circular_replay import CircularReplayBuffer
from hive.utils.advantage_computation_utils import AdvantageComputationFn


class OnPolicyReplayBuffer(CircularReplayBuffer):
    """An extension of the CircularReplayBuffer for on-policy learning algorithms"""

    def __init__(
        self,
        capacity: int = 10000,
        stack_size: int = 1,
        n_step: int = 1,
        gamma: float = 0.99,
        compute_advantage_fn: AdvantageComputationFn = None,
        observation_shape=(),
        observation_dtype=np.uint8,
        action_shape=(),
        action_dtype=np.int8,
        reward_shape=(),
        reward_dtype=np.float32,
        extra_storage_types=None,
    ):
        """Constructor for OnPolicyReplayBuffer.

        Args:
            capacity (int): Total number of observations that can be stored in the
                buffer
            stack_size (int): The number of frames to stack to create an observation.
            n_step (int): Horizon used to compute n-step return reward
            gamma (float): Discounting factor used to compute n-step return reward
            compute_advantage_fn (AdvantageComputationFn): Function used to compute the
                advantages.
            observation_shape: Shape of observations that will be stored in the buffer.
            observation_dtype: Type of observations that will be stored in the buffer.
                This can either be the type itself or string representation of the
                type. The type can be either a native python type or a numpy type. If
                a numpy type, a string of the form np.uint8 or numpy.uint8 is
                acceptable.
            action_shape: Shape of actions that will be stored in the buffer.
            action_dtype: Type of actions that will be stored in the buffer. Format is
                described in the description of observation_dtype.
            action_shape: Shape of actions that will be stored in the buffer.
            action_dtype: Type of actions that will be stored in the buffer. Format is
                described in the description of observation_dtype.
            reward_shape: Shape of rewards that will be stored in the buffer.
            reward_dtype: Type of rewards that will be stored in the buffer. Format is
                described in the description of observation_dtype.
            extra_storage_types (dict): A dictionary describing extra items to store
                in the buffer. The mapping should be from the name of the item to a
                (type, shape) tuple.
        """
        if extra_storage_types is None:
            extra_storage_types = dict()
        extra_storage_types.update(
            {
                "values": (np.float32, ()),
                "returns": (np.float32, ()),
                "advantages": (np.float32, ()),
                "logprob": (np.float32, ()),
            }
        )
        super().__init__(
            capacity + stack_size - 1,
            stack_size,
            n_step,
            gamma,
            observation_shape,
            observation_dtype,
            action_shape,
            action_dtype,
            reward_shape,
            reward_dtype,
            extra_storage_types,
        )
        self._compute_advantage_fn = compute_advantage_fn

    # Taken from https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_shared.py
    def compute_advantages(self, last_values):
        """Compute advantages using rewards and value estimates."""
        (
            self._storage["advantages"],
            self._storage["returns"],
        ) = self._compute_advantage_fn(
            self._storage["values"],
            last_values,
            self._storage["done"],
            self._storage["reward"],
            self._gamma,
        )

    def reset(self):
        """Resets the storage."""
        if self._stack_size > 1:
            saved_transitions = {
                k: self._storage[k][-(self._stack_size - 1) :]
                for k in self._storage.keys()
            }
            self._create_storage(self._capacity, self._specs)
            for k in self._storage.keys():
                self._storage[k][: (self._stack_size - 1)] = saved_transitions[k]
        else:
            self._create_storage(self._capacity, self._specs)
        self._cursor = self._stack_size - 1
        self._num_added = self._stack_size - 1

    def _find_valid_indices(self):
        """Filters invalid indices."""
        self._sample_cursor = 0
        self._valid_indices = self._filter_transitions(np.arange(self._capacity))
        self._valid_indices = self._rng.permutation(self._valid_indices)
        return len(self._valid_indices)

    def _sample_indices(self, batch_size):
        """Samples valid indices that can be used by the replay."""
        start = self._sample_cursor
        end = min(len(self._valid_indices), (self._sample_cursor + batch_size))
        indices = self._valid_indices[start:end]
        self._sample_cursor += batch_size
        return indices + self._stack_size - 1

    def sample(self, batch_size):
        valid_ind_size = self._find_valid_indices()
        num_batches = int(np.ceil(valid_ind_size / batch_size))
        for _ in range(num_batches):
            yield super().sample(batch_size)
