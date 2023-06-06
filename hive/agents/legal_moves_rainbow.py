from collections import deque

import numpy as np
import torch

from hive.agents.rainbow import RainbowDQNAgent
from hive.agents.utils import get_stacked_state
from hive.utils.loggers import logger


class LegalMovesRainbowAgent(RainbowDQNAgent):
    """A Rainbow agent which supports games with legal actions."""

    def create_q_networks(self, representation_net):
        """Creates the qnet and target qnet."""
        super().create_q_networks(representation_net)
        self._qnet = LegalMovesHead(self._qnet)
        self._target_qnet = LegalMovesHead(self._target_qnet)

    def preprocess_observation(self, observation, agent_traj_state):
        if agent_traj_state is None:
            observation_stack = deque(maxlen=self._stack_size - 1)
        else:
            observation_stack = agent_traj_state["observation_stack"]
        state, observation_stack = get_stacked_state(
            observation["observation"], observation_stack, self._stack_size
        )
        state = torch.tensor(state, device=self._device).unsqueeze(0).float()

        legal_moves = np.nonzero(observation["action_mask"])[0]
        encoded_legal_moves = torch.tensor(
            action_encoding(observation["action_mask"]), device=self._device
        ).float()
        return state, legal_moves, encoded_legal_moves, observation_stack

    def preprocess_update_info(self, update_info):
        preprocessed_update_info = {
            "observation": update_info["observation"]["observation"],
            "next_observation": update_info["next_observation"]["observation"],
            "action": update_info["action"],
            "reward": update_info["reward"],
            "terminated": update_info["terminated"],
            "truncated": update_info["truncated"],
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
    def act(self, observation, agent_traj_state, global_step):
        if self._training:
            if not self._learn_schedule.get_value():
                epsilon = 1.0
            elif not self._use_eps_greedy:
                epsilon = 0.0
            else:
                epsilon = self._epsilon_schedule(global_step)
            if self._log_schedule(global_step):
                logger.log_scalar("epsilon", epsilon, self._timescale)
        else:
            epsilon = self._test_epsilon

        (
            state,
            legal_moves,
            encoded_legal_moves,
            observation_stack,
        ) = self.preprocess_observation(observation, agent_traj_state)
        qvals = self._qnet(state, encoded_legal_moves)

        if self._rng.random() < epsilon:
            action = self._rng.choice(legal_moves)
        else:
            action = torch.argmax(qvals).item()

        if (
            self._training
            and self._log_schedule(global_step)
            and agent_traj_state is None
        ):
            logger.log_scalar("train_qval", torch.max(qvals), self._timescale)

        agent_traj_state = {"observation_stack": observation_stack}
        return action, agent_traj_state


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
