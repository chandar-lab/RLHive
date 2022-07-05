import numpy as np
import torch

from hive.agents.rainbow import RainbowDQNAgent


class LegalMovesRainbowAgent(RainbowDQNAgent):
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
        if self._training:
            if not self._learn_schedule.get_value():
                epsilon = 1.0
            elif not self._use_eps_greedy:
                epsilon = 0.0
            else:
                epsilon = self._epsilon_schedule.update()
            if self._logger.update_step(self._timescale):
                self._logger.log_scalar("epsilon", epsilon, self._timescale)
        else:
            epsilon = self._test_epsilon

        vectorized_observation = torch.tensor(
            np.expand_dims(observation["observation"], axis=0), device=self._device
        ).float()
        legal_moves_as_int = [
            i for i, x in enumerate(observation["action_mask"]) if x == 1
        ]
        encoded_legal_moves = torch.tensor(
            action_encoding(observation["action_mask"]), device=self._device
        ).float()
        qvals = self._qnet(vectorized_observation, encoded_legal_moves).cpu()

        if self._rng.random() < epsilon:
            action = np.random.choice(legal_moves_as_int).item()
        else:
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
    def __init__(self, base_network):
        super().__init__()
        self.base_network = base_network

    def forward(self, x, legal_moves):
        x = self.base_network(x)
        return x + legal_moves

    def dist(self, x, legal_moves):
        return self.base_network.dist(x)


def action_encoding(action_mask):
    encoded_action_mask = np.zeros(action_mask.shape)
    encoded_action_mask[action_mask == 0] = -np.inf
    return encoded_action_mask
