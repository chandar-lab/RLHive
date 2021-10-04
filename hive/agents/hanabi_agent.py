import numpy as np
import torch
from hive.agents.rainbow import RainbowDQNAgent


class HanabiAgent(RainbowDQNAgent):
    def create_q_networks(self, qnet):
        super().create_q_networks(qnet)
        self._qnet = HanabiHead(self._qnet)
        self._target_qnet = HanabiHead(self._target_qnet)

    def preprocess_update_info(self, update_info):
        return {
            "observation": np.array(
                update_info["observation"]["observation"], dtype=np.uint8
            ),
            "action": update_info["action"],
            "reward": update_info["reward"],
            "done": update_info["done"],
            "legal_moves": action_encoding(update_info["observation"]["action_mask"]),
        }

    def preprocess_update_batch(self, batch):
        return (
            (batch["observation"], batch["legal_moves"]),
            (batch["next_observation"], batch["next_legal_moves"]),
        )


def action_encoding(action_mask):
    encoded_action_mask = np.zeros(action_mask.shape)
    encoded_action_mask[action_mask == 0] = -np.inf
    return torch.tensor(encoded_action_mask)


class HanabiHead(torch.nn.Module):
    def __init__(self, base_network):
        super().__init__()
        self.base_network = base_network

    def forward(self, x, legal_moves):
        x = self.base_network(x)
        return x + legal_moves

    def dist(self, x, legal_moves):
        if not self._distributional:
            raise RuntimeError(
                "HanabiHead.dist() is not implemented when distributional is false"
            )
        return self.base_network.dist(x)
