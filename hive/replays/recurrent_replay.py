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
        hidden_spec=None,
        num_players_sharing_buffer: int = None,
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
        if hidden_spec is not None:
            if extra_storage_types is None:
                extra_storage_types = {}
            extra_storage_types.update(hidden_spec)
        self._hidden_spec = hidden_spec
        super().__init__(
            capacity=capacity,
            stack_size=max_seq_len,
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
            self.size() + self._stack_size + self._n_step - 1
        )
        elements = array[full_indices]
        return elements

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
        dones = self._get_from_storage(
            "done",
            indices - self._max_seq_len + 1,
            num_to_access=self._max_seq_len + self._n_step - 1,
        )
        terminated = self._get_from_storage(
            "done",
            indices - self._max_seq_len + 1,
            num_to_access=self._max_seq_len + self._n_step - 1,
        )

        if self._n_step == 1:
            is_terminal = dones
            trajectory_lengths = np.ones(batch_size)
        else:
            is_terminal = dones.any(axis=1).astype(int)
            terminated = terminated.any(axis=1).astype(int)
            trajectory_lengths = (
                np.argmax(dones.astype(bool), axis=1) + 1
            ) * is_terminal + self._n_step * (1 - is_terminal)
            is_terminal = dones[:, 1 : self._n_step - 1]
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
            elif key == "done":
                pass
            elif key == "terminated":
                batch["terminated"] = terminated
                batch["truncated"] = is_terminal - terminated
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
            elif key in self._hidden_spec:
                batch[key] = self._get_from_storage(
                    key,
                    indices - self._max_seq_len + 1,
                    num_to_access=1,
                )
                batch[f"next_{key}"] = self._get_from_storage(
                    key,
                    batch["indices"]
                    + trajectory_lengths
                    - self._max_seq_len
                    + 1,  # just return batch["indices"]
                    num_to_access=1,
                )
            else:
                batch[key] = self._get_from_storage(key, indices)

        batch["trajectory_lengths"] = trajectory_lengths
        batch["next_observation"] = self._get_from_storage(
            "observation",
            indices + trajectory_lengths - self._max_seq_len + 1,
            num_to_access=self._max_seq_len,
        )

        return batch
