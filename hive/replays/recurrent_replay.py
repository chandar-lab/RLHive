import os
import pickle

import numpy as np
from hive.replays.circular_replay import CircularReplayBuffer


class RecurrentReplayBuffer(CircularReplayBuffer):
    """
    First implementation of recurrent buffer without storing hidden states
    """

    def __init__(
        self,
        capacity: int = 10000,
        max_seq_len: int = 1,
        n_step: int = 1,
        gamma: float = 0.99,
        observation_shape=(),
        observation_dtype=np.uint8,
        action_shape=(),
        action_dtype=np.int8,
        reward_shape=(),
        reward_dtype=np.float32,
        extra_storage_types=None,
        num_players_sharing_buffer: int = None,
        rnn_type: str = "lstm",
        rnn_hidden_size: int = 0,
        store_hidden: bool = False,
    ):
        """Constructor for RecurrentReplayBuffer.

        Args:
            capacity (int): Total number of observations that can be stored in the
                buffer. Note, this is not the same as the number of transitions that
                can be stored in the buffer.
            max_seq_len (int): The number of consecutive transitions in a sequence
                sampled from an episode.
            n_step (int): Horizon used to compute n-step return reward
            gamma (float): Discounting factor used to compute n-step return reward
            observation_shape: Shape of observations that will be stored in the buffer.
            observation_dtype: Type of observations that will be stored in the buffer.
                This can either be the type itself or string representation of the
                type. The type can be either a native python type or a numpy type. If
                a numpy type, a string of the form np.uint8 or numpy.uint8 is
                acceptable.
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
            extra_storage_types = {}
        if store_hidden == True:
            extra_storage_types["hidden_state"] = (
                np.float32,
                (1, 1, rnn_hidden_size),
            )
            if rnn_type == "lstm":
                extra_storage_types["cell_state"] = (
                    np.float32,
                    (1, 1, rnn_hidden_size),
                )
            elif rnn_type != "gru":
                raise ValueError(
                    f"rnn_type is wrong. Expected either lstm or gru,"
                    f"received {rnn_type}."
                )
        super().__init__(
            capacity=capacity,
            stack_size=1,
            n_step=n_step,
            gamma=gamma,
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            action_shape=action_shape,
            action_dtype=action_dtype,
            reward_shape=reward_shape,
            reward_dtype=reward_dtype,
            extra_storage_types=extra_storage_types,
            num_players_sharing_buffer=num_players_sharing_buffer,
        )
        self._max_seq_len = max_seq_len
        self._rnn_type = rnn_type
        self._rnn_hidden_size = rnn_hidden_size
        self._store_hidden = store_hidden

    def size(self):
        """Returns the number of transitions stored in the buffer."""
        return max(
            min(self._num_added, self._capacity) - self._max_seq_len - self._n_step + 1,
            0,
        )

    def add(self, observation, action, reward, done, **kwargs):
        """Adds a transition to the buffer.
        The required components of a transition are given as positional arguments. The
        user can pass additional components to store in the buffer as kwargs as long as
        they were defined in the specification in the constructor.
        """

        if self._episode_start:
            self._pad_buffer(self._max_seq_len - 1)
            self._episode_start = False
        transition = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "done": done,
            "mask": 1,
        }
        transition.update(kwargs)
        for key in self._specs:
            obj_type = (
                transition[key].dtype
                if hasattr(transition[key], "dtype")
                else type(transition[key])
            )
            if not np.can_cast(obj_type, self._specs[key][0], casting="same_kind"):
                raise ValueError(
                    f"Key {key} has wrong dtype. Expected {self._specs[key][0]},"
                    f"received {type(transition[key])}."
                )
        if self._num_players_sharing_buffer is None:
            self._add_transition(**transition)
        else:
            self._episode_storage[kwargs["agent_id"]].append(transition)
            if done:
                for transition in self._episode_storage[kwargs["agent_id"]]:
                    self._add_transition(**transition)
                self._episode_storage[kwargs["agent_id"]] = []

        if done:
            self._episode_start = True

    def _get_from_array(self, array, indices, num_to_access=1):
        """Retrieves consecutive elements in the array, wrapping around if necessary.
        If more than 1 element is being accessed, the elements are concatenated along
        the first dimension.
        Args:
            array: array to access from
            indices: starts of ranges to access from
            num_to_access: how many consecutive elements to access
        """
        full_indices = np.indices((indices.shape[0], num_to_access))[1]
        full_indices = (full_indices + np.expand_dims(indices, axis=1)) % (
            self.size() + self._max_seq_len + self._n_step - 1
        )
        elements = array[full_indices]
        elements = elements.reshape(indices.shape[0], -1, *elements.shape[2:])
        return elements

    def _get_from_storage(self, key, indices, num_to_access=1):
        """Gets values from storage.
        Args:
            key: The name of the component to retrieve.
            indices: This can be a single int or a 1D numpyp array. The indices are
                adjusted to fall within the current bounds of the buffer.
            num_to_access: how many consecutive elements to access
        """
        if not isinstance(indices, np.ndarray):
            indices = np.array([indices])
        if num_to_access == 0:
            return np.array([])
        elif num_to_access == 1:
            return self._storage[key][
                indices % (self.size() + self._max_seq_len + self._n_step - 1)
            ]
        else:
            return self._get_from_array(
                self._storage[key], indices, num_to_access=num_to_access
            )

    def _sample_indices(self, batch_size):
        """Samples valid indices that can be used by the replay."""
        indices = np.array([], dtype=np.int32)
        while len(indices) < batch_size:
            start_index = (
                self._rng.integers(self.size(), size=batch_size - len(indices))
                + self._cursor
            )
            start_index = self._filter_transitions(start_index)
            indices = np.concatenate([indices, start_index])
        return indices + self._max_seq_len - 1

    def _filter_transitions(self, indices):
        """Filters invalid indices."""
        if self._max_seq_len == 1:
            return indices
        done = self._get_from_storage("done", indices, self._max_seq_len - 1)
        done = done.astype(bool)
        if self._max_seq_len == 2:
            indices = indices[~done]
        else:
            indices = indices[~done.any(axis=1)]
        return indices

    def sample(self, batch_size):
        """Sample transitions from the buffer. For a given transition, if it's
        done is True, the next_observation value should not be taken to have any
        meaning.

        Args:
            batch_size (int): Number of transitions to sample.
        """
        if self._num_added < self._max_seq_len + self._n_step:
            raise ValueError("Not enough transitions added to the buffer to sample")
        indices = self._sample_indices(batch_size)
        batch = {}
        batch["indices"] = indices
        terminals = self._get_from_storage(
            "done",
            indices - self._max_seq_len + 1,
            num_to_access=self._max_seq_len + self._n_step - 1,
        )

        if self._n_step == 1:
            is_terminal = terminals
            trajectory_lengths = np.ones(batch_size)
        else:
            is_terminal = terminals.any(axis=1).astype(int)
            trajectory_lengths = (
                np.argmax(terminals.astype(bool), axis=1) + 1
            ) * is_terminal + self._n_step * (1 - is_terminal)
            is_terminal = terminals[:, 1 : self._n_step - 1]
        trajectory_lengths = trajectory_lengths.astype(np.int64)

        for key in self._specs:
            if key == "observation":
                batch[key] = self._get_from_storage(
                    "observation",
                    indices - self._max_seq_len + 1,
                    num_to_access=self._max_seq_len,
                )
            elif key == "action":
                batch[key] = self._get_from_storage(
                    "action",
                    indices - self._max_seq_len + 1,
                    num_to_access=self._max_seq_len,
                )
            elif key == "hidden_state":
                batch[key] = self._get_from_storage(
                    "hidden_state",
                    indices - self._max_seq_len + 1,
                    num_to_access=self._max_seq_len,
                )
            elif key == "cell_state":
                batch[key] = self._get_from_storage(
                    "cell_state",
                    indices - self._max_seq_len + 1,
                    num_to_access=self._max_seq_len,
                )
            elif key == "done":
                batch["done"] = is_terminal
            elif key == "reward":
                rewards = self._get_from_storage(
                    "reward",
                    indices - self._max_seq_len + 1,
                    num_to_access=self._max_seq_len + self._n_step - 1,
                )
                if self._max_seq_len + self._n_step - 1 == 1:
                    rewards = np.expand_dims(rewards, 1)

                if self._n_step == 1:
                    rewards = rewards * np.expand_dims(self._discount, axis=0)

                elif self._n_step > 1:
                    idx = np.arange(rewards.shape[1] - self._n_step + 1)[
                        :, None
                    ] + np.arange(
                        self._n_step
                    )  # (S-N+1) x N
                    rewards = rewards[:, idx]  # B x (S-N+1) x N
                    # Creating a vectorized sliding window to calculate
                    # discounted returns for every element in the sequence.
                    # Equivalent to
                    # np.sum(rewards * self._discount[None, None, :], axis=2)
                    disc_rewards = np.einsum("ijk,k->ij", rewards, self._discount)
                    rewards = disc_rewards

                batch["reward"] = rewards
            elif key == "mask":
                batch[key] = self._get_from_storage(
                    "mask",
                    indices - self._max_seq_len + 1,
                    num_to_access=self._max_seq_len,
                )
            else:
                batch[key] = self._get_from_storage(key, indices)

        batch["trajectory_lengths"] = trajectory_lengths
        batch["next_observation"] = self._get_from_storage(
            "observation",
            indices + trajectory_lengths - self._max_seq_len + 1,
            num_to_access=self._max_seq_len,
        )

        if self._store_hidden == True:
            batch["next_hidden_state"] = self._get_from_storage(
                "hidden_state",
                batch["indices"]
                + batch["trajectory_lengths"]
                - self._max_seq_len
                + 1,  # just return batch["indices"]
                num_to_access=self._max_seq_len,
            )
            if self._rnn_type == "lstm":
                batch["next_cell_state"] = self._get_from_storage(
                    "cell_state",
                    batch["indices"]
                    + batch["trajectory_lengths"]
                    - self._max_seq_len
                    + 1,
                    num_to_access=self._max_seq_len,
                )

        return batch
