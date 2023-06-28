from collections import deque
from functools import partial
from typing import Optional

import gymnasium as gym
import numpy as np
import torch

from hive.agents.dqn.rainbow import RainbowDQNAgent
from hive.replays import PrioritizedReplayBuffer
from hive.replays.replay_buffer import Alignment, ReplayItemSpec
from hive.types import Creates, Partial, default
from hive.utils.loggers import logger
from hive.utils.np_utils import roll_state
from hive.utils.schedule import Schedule
from hive.utils.torch_utils import ModuleInitFn
from hive.utils.utils import LossFn


class LegalMovesRainbowAgent(RainbowDQNAgent):
    """A Rainbow agent which supports games with legal actions."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        representation_net: Creates[torch.nn.Module],
        stack_size: int = 1,
        optimizer_fn: Optional[Creates[torch.optim.Optimizer]] = None,
        loss_fn: Optional[Creates[LossFn]] = None,
        init_fn: Optional[Partial[ModuleInitFn]] = None,
        id=0,
        replay_buffer: Optional[Creates[PrioritizedReplayBuffer]] = None,
        discount_rate: float = 0.99,
        n_step: int = 1,
        grad_clip: Optional[float] = None,
        reward_clip: Optional[float] = None,
        update_period_schedule: Optional[Creates[Schedule[bool]]] = None,
        target_net_soft_update: bool = False,
        target_net_update_fraction: float = 0.05,
        target_net_update_schedule: Optional[Creates[Schedule[bool]]] = None,
        epsilon_schedule: Optional[Creates[Schedule[float]]] = None,
        test_epsilon: float = 0.001,
        min_replay_history: int = 5000,
        batch_size: int = 32,
        device="cpu",
        log_frequency: int = 100,
        noisy: bool = True,
        std_init: float = 0.5,
        use_eps_greedy: bool = False,
        double: bool = True,
        dueling: bool = True,
        distributional: bool = True,
        v_min: float = 0,
        v_max: float = 200,
        atoms: int = 51,
    ):
        replay_buffer = default(replay_buffer, PrioritizedReplayBuffer)
        replay_buffer = partial(
            replay_buffer,
            extra_storage_specs={
                "action_mask": ReplayItemSpec.create(
                    (int(action_space.n),), np.float32, return_next=True
                )
            },
        )
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            representation_net=representation_net,
            stack_size=stack_size,
            optimizer_fn=optimizer_fn,
            loss_fn=loss_fn,
            init_fn=init_fn,
            id=id,
            replay_buffer=replay_buffer,
            discount_rate=discount_rate,
            n_step=n_step,
            grad_clip=grad_clip,
            reward_clip=reward_clip,
            update_period_schedule=update_period_schedule,
            target_net_soft_update=target_net_soft_update,
            target_net_update_fraction=target_net_update_fraction,
            target_net_update_schedule=target_net_update_schedule,
            epsilon_schedule=epsilon_schedule,
            test_epsilon=test_epsilon,
            min_replay_history=min_replay_history,
            batch_size=batch_size,
            device=device,
            log_frequency=log_frequency,
            noisy=noisy,
            std_init=std_init,
            use_eps_greedy=use_eps_greedy,
            double=double,
            dueling=dueling,
            distributional=distributional,
            v_min=v_min,
            v_max=v_max,
            atoms=atoms,
        )

    def create_networks(self, representation_net):
        """Creates the qnet and target qnet."""
        super().create_networks(representation_net)
        self._qnet = LegalMovesHead(self._qnet)
        self._target_qnet = LegalMovesHead(self._target_qnet)

    def preprocess_observation(self, observation, agent_traj_state):
        state, observation_stack = super().preprocess_observation(
            observation["observation"], agent_traj_state
        )

        legal_moves = np.nonzero(observation["action_mask"])[0]
        encoded_legal_moves = torch.tensor(
            action_encoding(observation["action_mask"]), device=self._device
        ).float()
        return (*state, encoded_legal_moves), legal_moves, observation_stack

    def preprocess_update_info(self, update_info):
        preprocessed_update_info = {
            "observation": update_info["observation"]["observation"],
            "next_observation": update_info["next_observation"]["observation"],
            "action": update_info["action"],
            "reward": update_info["reward"],
            "terminated": update_info["terminated"],
            "truncated": update_info["truncated"],
            "action_mask": action_encoding(update_info["observation"]["action_mask"]),
            "source": update_info["source"],
        }
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
            if not self._learn_schedule(global_step):
                epsilon = 1.0
            elif not self._use_eps_greedy:
                epsilon = 0.0
            else:
                epsilon = self._epsilon_schedule(global_step)
            if self._log_schedule(global_step):
                logger.log_scalar("epsilon", epsilon, self.id)
        else:
            epsilon = self._test_epsilon
        (
            state,
            legal_moves,
            observation_stack,
        ) = self.preprocess_observation(observation, agent_traj_state)
        qvals = self._qnet(*state)

        if self._rng.random() < epsilon:
            action = self._rng.choice(legal_moves)
        else:
            action = torch.argmax(qvals).item()

        if (
            self._training
            and self._log_schedule(global_step)
            and agent_traj_state is None
        ):
            logger.log_scalar("train_qval", torch.max(qvals), self.id)

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
