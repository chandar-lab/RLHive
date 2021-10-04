import copy
from typing import Tuple, Callable
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from hive.agents.rainbow import RainbowDQNAgent
from hive.agents.qnets.base import FunctionApproximator
from hive.replays.replay_buffer import BaseReplayBuffer
from hive.utils.logging import Logger
from hive.utils.schedule import Schedule
from hive.utils.utils import OptimizerFn


class HanabiRainbowAgent(RainbowDQNAgent):
    """A Hanabi agent implementing the Rainbow algorithm."""

    def __init__(
        self,
        qnet: FunctionApproximator,
        obs_dim: Tuple,
        act_dim: int,
        v_min: str = 0,
        v_max: str = 200,
        atoms: str = 51,
        optimizer_fn: OptimizerFn = None,
        id: str = 0,
        replay_buffer: BaseReplayBuffer = None,
        discount_rate: float = 0.99,
        n_step: int = 1,
        grad_clip: float = None,
        reward_clip: float = None,
        target_net_soft_update: bool = False,
        target_net_update_fraction: float = 0.05,
        target_net_update_schedule: Schedule = None,
        update_period_schedule: Schedule = None,
        epsilon_schedule: Schedule = None,
        learn_schedule: Schedule = None,
        seed: int = 42,
        batch_size: int = 32,
        device: str = "cpu",
        logger: Logger = None,
        log_frequency: int = 100,
        noisy=True,
        std_init=0.5,
        double: bool = True,
        dueling: bool = True,
        distributional: bool = True,
        use_eps_greedy: bool = False,
    ):
        """
        Args:
        """
        super().__init__(
            qnet,
            obs_dim,
            act_dim,
            v_min=v_min,
            v_max=v_max,
            atoms=atoms,
            optimizer_fn=optimizer_fn,
            id=id,
            replay_buffer=replay_buffer,
            discount_rate=discount_rate,
            n_step=n_step,
            grad_clip=grad_clip,
            reward_clip=reward_clip,
            target_net_soft_update=target_net_soft_update,
            target_net_update_fraction=target_net_update_fraction,
            target_net_update_schedule=target_net_update_schedule,
            update_period_schedule=update_period_schedule,
            epsilon_schedule=epsilon_schedule,
            learn_schedule=learn_schedule,
            seed=seed,
            batch_size=batch_size,
            device=device,
            logger=logger,
            log_frequency=log_frequency,
            noisy=noisy,
            std_init=std_init,
            double=double,
            dueling=dueling,
            distributional=distributional,
            use_eps_greedy=use_eps_greedy,
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

        vectorized_observation = (
            torch.tensor(np.expand_dims(observation["observation"], axis=0))
            .to(self._device)
            .float()
        )
        legal_moves_as_int = [
            i for i, x in enumerate(observation["action_mask"]) if x == 1
        ]
        encoded_legal_moves = action_encoding(observation["action_mask"])
        qvals = self._qnet(
            vectorized_observation, torch.tensor(encoded_legal_moves).to(self._device)
        ).cpu()

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
            "action_mask": np.array(
                action_encoding(update_info["observation"]["action_mask"]),
                dtype=np.uint8,
            ),
        }

    def preprocess_update_batch(self, batch):
        for key in batch:
            batch[key] = torch.tensor(batch[key]).to(self._device)
        return (
            (batch["observation"], batch["action_mask"]),
            (batch["next_observation"], batch["next_action_mask"]),
        )


def action_encoding(action_mask):
    encoded_action_mask = np.zeros(action_mask.shape)
    encoded_action_mask[action_mask == 0] = -np.inf
    return encoded_action_mask


class HanabiHead(torch.nn.Module):
    def __init__(self, base_network):
        super().__init__()
        self.base_network = base_network

    def forward(self, x, legal_moves):
        x = self.base_network(x)
        return x + legal_moves

    def dist(self, x, legal_moves):
        return self.base_network.dist(x)
