import numpy as np
from hive.replays.circular_replay import CircularReplayBuffer


class PPOReplayBuffer(CircularReplayBuffer):
    """An extension of the CircularReplayBuffer for on-policy learning algorithms"""

    def __init__(
        self,
        capacity: int = 10000,
        stack_size: int = 1,
        n_step: int = 1,
        gamma: float = 0.99,
        use_gae: bool = True,
        gae_lambda: float = 0.95,
        observation_shape=(),
        observation_dtype=np.uint8,
        action_shape=(),
        action_dtype=np.int8,
        reward_shape=(),
        reward_dtype=np.float32,
        extra_storage_types=None,
        num_players_sharing_buffer: int = None,
    ):
        """Constructor for PPOReplayBuffer.

        Args:
            capacity (int): Total number of observations that can be stored in the buffer
            stack_size (int): The number of frames to stack to create an observation.
            n_step (int): Horizon used to compute n-step return reward
            gamma (float): Discounting factor used to compute n-step return reward
            use_gae (bool): Whether to use generalised advantage estimates for calculating returns
            gae_lambda (float): Discouting factor used to compute generalised advantage estimation
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
            num_players_sharing_buffer (int): Number of agents that share their
                buffers. It is used for self-play.
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
            num_players_sharing_buffer,
        )
        self._use_gae = use_gae
        self._gae_lambda = gae_lambda

    # Taken from https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_shared.py
    def compute_advantages(self, values):
        """Compute advantages using rewards and value estimates."""
        if self._use_gae:
            lastgaelam = 0
            for t in reversed(range(self._capacity)):
                nextvalues = (
                    values
                    if t == self._capacity - 1
                    else self._storage["values"][t + 1]
                )
                nextnonterminal = 1.0 - self._storage["done"][t]
                delta = (
                    self._storage["reward"][t]
                    + self._gamma * nextvalues * nextnonterminal
                    - self._storage["values"][t]
                )
                self._storage["advantages"][t] = lastgaelam = (
                    delta
                    + self._gamma * self._gae_lambda * nextnonterminal * lastgaelam
                )
            self._storage["returns"] = (
                self._storage["advantages"] + self._storage["values"]
            )
        else:
            for t in reversed(range(self._capacity)):
                next_return = (
                    values
                    if t == self._capacity - 1
                    else self._storage["return"][t + 1]
                )
                nextnonterminal = 1.0 - self._storage["done"][t]
                self._storage["returns"][t] = (
                    self._storage["reward"][t]
                    + self._gamma * nextnonterminal * next_return
                )
            self._storage["advantages"] = (
                self._storage["returns"] - self._storage["values"]
            )

    def reset(self):
        """Resets the storage."""
        if self._stack_size > 1:
            transitions = {
                k: self._storage[k][-(self._stack_size - 1) :]
                for k in self._storage.keys()
            }
            self._create_storage(self._capacity, self._specs)
            for k in self._storage.keys():
                self._storage[k][: (self._stack_size - 1)] = transitions[k]
            self._episode_start = transitions["done"][-1]
        else:
            self._create_storage(self._capacity, self._specs)
            self._episode_start = True
        self._cursor = self._stack_size - 1
        self._num_added = self._stack_size - 1

    def sample(self, batch_size):
        """Sample transitions from the buffer. For a given transition, if it's
        done is True, the next_observation value should not be taken to have any
        meaning.

        Args:
            batch_size (int): Number of transitions to sample.
        """
        if self._num_added < self._stack_size + self._n_step:
            raise ValueError("Not enough transitions added to the buffer to sample")

        all_indices = self._filter_transitions(np.arange(self._capacity))
        all_indices = self._rng.permutation(all_indices)
        for i in range(0, len(all_indices), batch_size):
            indices = all_indices[i : i + batch_size]
            batch = {}
            batch["indices"] = indices
            terminals = self._get_from_storage("done", indices, self._n_step)
            if self._n_step == 1:
                is_terminal = terminals
                trajectory_lengths = np.ones(terminals.shape[0])
            else:
                is_terminal = terminals.any(axis=1).astype(int)
                trajectory_lengths = (
                    np.argmax(terminals.astype(bool), axis=1) + 1
                ) * is_terminal + self._n_step * (1 - is_terminal)
            trajectory_lengths = trajectory_lengths.astype(np.int64)

            for key in self._specs:
                if key == "observation":
                    batch[key] = self._get_from_storage(
                        "observation",
                        indices - self._stack_size + 1,
                        num_to_access=self._stack_size,
                    )
                elif key == "done":
                    batch["done"] = is_terminal
                elif key == "reward":
                    rewards = self._get_from_storage("reward", indices, self._n_step)
                    if self._n_step == 1:
                        rewards = np.expand_dims(rewards, 1)
                    rewards = rewards * np.expand_dims(self._discount, axis=0)

                    # Mask out rewards past trajectory length
                    mask = np.expand_dims(trajectory_lengths, 1) > np.arange(
                        self._n_step
                    )
                    rewards = np.sum(rewards * mask, axis=1)
                    batch["reward"] = rewards
                else:
                    batch[key] = self._get_from_storage(key, indices)

            batch["trajectory_lengths"] = trajectory_lengths
            batch["next_observation"] = self._get_from_storage(
                "observation",
                indices + trajectory_lengths - self._stack_size + 1,
                num_to_access=self._stack_size,
            )
            yield batch