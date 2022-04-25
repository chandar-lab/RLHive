from typing import Dict, Tuple

import numpy as np
import torch 

from hive.replays.recurrent_replay import RecurrentReplayBuffer


class RecurrentHiddenReplayBuffer(RecurrentReplayBuffer):
    """ 
    Storing hidden state and cell state in the buffer for DRQN
    """

    def __init__(
        self,
        capacity: int,
        max_seq_len: int = 1,
        n_step: int = 1,
        gamma: float = 0.9,
        observation_shape: Tuple = (),
        observation_dtype: type = np.uint8,
        action_shape: Tuple = (),
        action_dtype: type = np.int8,
        reward_shape: Tuple = (),
        reward_dtype: type = np.float32,
        extra_storage_types: Dict = None,
        num_players_sharing_buffer: int = None,
        
    ):
        if extra_storage_types is None:
            extra_storage_types = {}
        extra_storage_types["hidden_state"] =   (np.float32, (1,1,128) )
        extra_storage_types["cell_state"]   =   (np.float32, (1,1,128) )
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
        )
    
    def sample(self, batch_size):
        """Sample transitions from the buffer.
        Adding next_action_mask to the batch for environments with legal moves.
        """

        batch = super().sample(batch_size)

        batch["next_hidden_state"] = self._get_from_storage(
            "hidden_state",
            batch["indices"] + batch["trajectory_lengths"] - self._max_seq_len + 1, #just return batch["indices"]
            num_to_access=self._max_seq_len,
        )

        batch["next_cell_state"] = self._get_from_storage(
            "cell_state",
            batch["indices"] + batch["trajectory_lengths"] - self._max_seq_len + 1,
            num_to_access=self._max_seq_len,
        )

        return batch