import numpy as np
from typing import Optional, MutableMapping, Mapping
from hive.replays.circular_replay import CircularReplayBuffer
from hive.utils.advantage import AdvantageComputationFn, compute_standard_advantages
from hive.utils.registry import OCreates, default, Partial

from hive.replays.replay_buffer import ReplayItemSpec, BaseReplayBuffer
from hive.utils.utils import seeder


class OnPolicyReplayBuffer(BaseReplayBuffer):
    """An extension of the CircularReplayBuffer for on-policy learning algorithms"""

    def __init__(
        self,
        num_sources: int = 1,
        steps_per_update: int = 10000,
        n_step: int = 1,
        gamma: float = 0.99,
        compute_advantage_fn: Optional[Partial[AdvantageComputationFn]] = None,
        observation_spec: ReplayItemSpec = ReplayItemSpec.create((), np.uint8),
        action_spec: ReplayItemSpec = ReplayItemSpec.create((), np.int8),
        reward_spec: ReplayItemSpec = ReplayItemSpec.create((), np.float32),
        extra_storage_specs: Optional[MutableMapping[str, ReplayItemSpec]] = None,
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
            extra_storage_specs (dict): A dictionary describing extra items to store
                in the buffer. The mapping should be from the name of the item to a
                (type, shape) tuple.
        """
        self._specs = {
            "observation": observation_spec,
            "next_observation": observation_spec,
            "action": action_spec,
            "reward": reward_spec,
            "done": ReplayItemSpec.create((), np.uint8),
            "terminated": ReplayItemSpec.create((), np.uint8),
            "value": ReplayItemSpec.create((), np.float32),
            "logprob": ReplayItemSpec.create((), np.float32),
        }
        if extra_storage_specs is None:
            extra_storage_specs = dict()
        self._specs.update(extra_storage_specs)
        self._storage = self._create_storage(steps_per_update)
        self._steps_per_update = steps_per_update
        self._gamma = gamma
        self._n_step = n_step
        self._rng = np.random.default_rng(seed=seeder.get_new_seed("replay"))
        self._cursors = np.zeros(num_sources, dtype=np.int64)
        self._compute_advantage_fn = default(
            compute_advantage_fn, compute_standard_advantages
        )
        self._num_sources = num_sources
        self._num_added = 0

    def size(self):
        return self._num_added

    def _create_storage(self, capacity):
        """Creates the storage for the replay buffer."""
        return {
            k: np.empty((capacity, self._num_sources) + v.shape, dtype=v.dtype)
            for k, v in self._specs.items()
        }

    def reset(self):
        """Resets the storage."""
        self._storage = self._create_storage(self._steps_per_update)
        self._cursors = np.zeros_like(self._cursors)
        self._num_added = 0

    def add_transitions(
        self,
        sources: np.ndarray,
        **transitions: Mapping[str, np.ndarray],
    ):
        for key in self._specs:
            obj_type = (
                transitions[key].dtype  # type: ignore
                if hasattr(transitions[key], "dtype")
                else type(transitions[key])
            )
            if not np.can_cast(obj_type, self._specs[key].dtype, casting="same_kind"):
                raise ValueError(
                    f"Key {key} has wrong dtype. Expected {self._specs[key].dtype},"
                    f"received {type(transitions[key])}."
                )
        cursors = self._cursors[sources]
        for k, v in transitions.items():
            if k in self._specs:
                self._storage[k][cursors][sources] = v
        self._cursors[sources] += 1
        self._num_added += len(sources)

    def compute_advantages(self, last_values, sources):
        max_steps = np.max(self._cursors)
        values = self._storage["value"][:max_steps]
        next_values = np.zeros(self._num_sources, dtype=np.float32)
        next_values[sources] = last_values
        terminated = self._storage["terminated"][:max_steps]
        rewards = self._storage["reward"][:max_steps]
        advantages, returns = self._compute_advantage_fn(
            values, next_values, terminated, rewards, self._gamma
        )
        self._storage["advantage"][:max_steps] = advantages
        self._storage["return"][:max_steps] = returns

    def sample(self, batch_size):
        full_batch = self._make_full_batch()
        inds = np.arange(len(full_batch["observation"]))
        self._rng.shuffle(inds)
        split_inds = np.array_split(inds, batch_size)
        for indicies in split_inds:
            yield {k: full_batch[k][indicies] for k in full_batch}

    def _make_full_batch(self):
        """Creates a batch of data from the replay buffer."""
        batch = {}
        for key, array in self._storage.items():
            batch[key] = np.concatenate(
                tuple(array[i][: self._cursors[i]] for i in range(self._num_sources)),
                axis=0,
            )

        return batch
