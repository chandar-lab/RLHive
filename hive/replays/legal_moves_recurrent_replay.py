from typing import Dict, Tuple

import numpy as np

from hive.replays.recurrent_replay import RecurrentReplayBuffer


class LegalMovesRecurrentBuffer(RecurrentReplayBuffer):
    """A Prioritized Replay buffer for the games like Hanabi with legal moves
    which need to add next_action_mask to the batch.
    """

    def __init__(
        self,
        capacity: int,
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
        action_dim: int = None,
        num_players_sharing_buffer: int = None,
        rnn_type: str = "lstm",
        rnn_hidden_size: int = 128,
        store_hidden: bool = False,
    ):
        if extra_storage_types is None:
            extra_storage_types = {}
        extra_storage_types["action_mask"] = (np.float, [action_dim])
        super().__init__(
            capacity=capacity,
            max_seq_len=max_seq_len,
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
            rnn_type=rnn_type,
            rnn_hidden_size=rnn_hidden_size,
            store_hidden=store_hidden,
        )

    def sample(self, batch_size):
        """Sample transitions from the buffer.
        Adding next_action_mask to the batch for environments with legal moves.
        """
        batch = super().sample(batch_size)
        batch["action_mask"] = self._get_from_storage(
            "action_mask",
            batch["indices"] - self._max_seq_len + 1,
            num_to_access=self._max_seq_len,
        )

        batch["next_action_mask"] = self._get_from_storage(
            "action_mask",
            batch["indices"] + batch["trajectory_lengths"] - self._max_seq_len + 1,
            num_to_access=self._max_seq_len,
        )

        return batch
