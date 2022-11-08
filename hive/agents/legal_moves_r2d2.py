import numpy as np
import torch

from hive.agents.r2d2 import R2D2Agent
from hive.agents.legal_moves_rainbow import action_encoding


class LegalMovesR2D2Agent(R2D2Agent):
    """A Rainbow agent which supports games with legal actions."""

    def create_q_networks(self, representation_net):
        """Creates the qnet and target qnet."""
        super().create_q_networks(representation_net)
        self._qnet = LegalMovesHead(self._qnet)
        self._target_qnet = LegalMovesHead(self._target_qnet)

    def preprocess_update_info(self, update_info):
        preprocessed_update_info = {
            "observation": update_info["observation"]["observation"],
            "action": update_info["action"],
            "reward": update_info["reward"],
            "done": update_info["done"],
            "action_mask": action_encoding(update_info["observation"]["action_mask"]),
        }
        if "agent_id" in update_info:
            preprocessed_update_info["agent_id"] = int(update_info["agent_id"])
        return preprocessed_update_info

    def preprocess_update_batch(self, batch):
        for key in batch:
            batch[key] = torch.tensor(batch[key], device=self._device)
        return (
            (batch["observation"], batch["action_mask"]),
            (batch["next_observation"], batch["next_action_mask"]),
            batch,
        )

    @torch.no_grad()
    def act(self, observation):
        """Returns the action for the agent. If in training mode, follows an epsilon
        greedy policy. Otherwise, returns the action with the highest Q-value.

        Args:
            observation: The current observation.
        """

        # Reset hidden state if it is episode beginning.
        if self._state["episode_start"]:
            self._hidden_state = self._qnet.init_hidden(batch_size=1)

        # Determine and log the value of epsilon
        if self._training:
            if not self._learn_schedule.get_value():
                epsilon = 1.0
            else:
                epsilon = self._epsilon_schedule.update()
            if self._logger.update_step(self._timescale):
                self._logger.log_scalar("epsilon", epsilon, self._timescale)
        else:
            epsilon = self._test_epsilon

        # Sample action. With epsilon probability choose random action,
        # otherwise select the action with the highest q-value.
        # Insert batch_size and sequence_len dimensions to observation
        vectorized_observation = torch.tensor(
            np.expand_dims(observation["observation"], axis=0), device=self._device
        ).float()
        encoded_legal_moves = torch.tensor(
            action_encoding(observation["action_mask"]), device=self._device
        ).float()
        qvals, self._hidden_state = self._qnet(
            vectorized_observation, encoded_legal_moves, self._hidden_state
        )

        if self._rng.random() < epsilon:
            action = np.random.choice(legal_moves_as_int).item()
        else:
            # Note: not explicitly handling the ties
            action = torch.argmax(qvals).item()

        if (
            self._training
            and self._logger.should_log(self._timescale)
            and self._state["episode_start"]
        ):
            self._logger.log_scalar("train_qval", torch.max(qvals), self._timescale)
            self._state["episode_start"] = False
        return action


class LegalMovesHead(torch.nn.Module):
    def __init__(self, qnet):
        super().__init__()
        self._legal_qnet = qnet
        self.base_network = qnet.base_network

    def forward(self, x, legal_moves, hidden_state=None):
        x, hidden_state = self._legal_qnet(x, hidden_state)
        legal_moves = legal_moves.view(-1, legal_moves.shape[-1])

        return x + legal_moves, hidden_state
