import copy
from functools import partial
from typing import Tuple, Callable
import numpy as np
import torch
from hive.agents.rainbow import RainbowDQNAgent
from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.qnet_heads_hanabi import (
    HanabiDistributionalNetwork,
    HanabiDQNNetwork,
    HanabiDuelingNetwork,
)
from hive.agents.qnets.noisy_linear import NoisyLinear
from hive.agents.qnets.utils import calculate_output_dim
from hive.replays import PrioritizedReplayBuffer
from hive.replays.replay_buffer import BaseReplayBuffer
from hive.utils.logging import Logger
from hive.utils.schedule import Schedule
from hive.utils.utils import OptimizerFn
from hive.agents.agent import Agent


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

    def create_q_networks(self, qnet, device):
        network = qnet(self._obs_dim)
        network_output_dim = np.prod(calculate_output_dim(network, self._obs_dim))

        # Use NoisyLinear when creating output heads if noisy is true
        linear_fn = (
            partial(NoisyLinear, std_init=self._std_init)
            if self._noisy
            else torch.nn.Linear
        )

        # Set up Dueling heads
        if self._dueling:
            network = HanabiDuelingNetwork(
                network, network_output_dim, self._act_dim, linear_fn, self._atoms
            )
        else:
            network = HanabiDQNNetwork(
                network, network_output_dim, self._act_dim * self._atoms, linear_fn
            )

        # Set up DistributionalNetwork wrapper if distributional is true
        if self._distributional:
            self._qnet = HanabiDistributionalNetwork(
                network, self._act_dim, self._v_min, self._v_max, self._atoms
            )
        else:
            self._qnet = network
        self._qnet.to(device=device)
        self._target_qnet = copy.deepcopy(self._qnet).requires_grad_(False)

    def target_projection(self, next_observation, next_legal_moves, reward, done):
        """Project distribution of target Q-values."""
        next_observation = next_observation.float()
        reward = reward.reshape(-1, 1)
        not_done = 1 - done.reshape(-1, 1)
        batch_size = next_observation.size(0)
        next_action = self._target_qnet(next_observation, next_legal_moves).argmax(1)
        next_dist = self._target_qnet.dist(next_observation, next_legal_moves)
        next_dist = next_dist[torch.arange(batch_size), next_action]

        dist_supports = reward + not_done * self._discount_rate * self._supports
        dist_supports = dist_supports.clamp(min=self._v_min, max=self._v_max)
        dist_supports = dist_supports.unsqueeze(1)
        dist_supports.tile([1, self._atoms, 1])
        projected_supports = self._supports.tile([batch_size, 1]).unsqueeze(2)

        delta = float(self._v_max - self._v_min) / (self._atoms - 1)
        quotient = 1 - (torch.abs(dist_supports - projected_supports) / delta)
        quotient = quotient.clamp(min=0, max=1)

        projection = torch.sum(quotient * next_dist.unsqueeze(1), dim=2)
        return projection

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
            epsilon = 0

        vectorized_observation = (
            torch.tensor(np.expand_dims(observation["vectorized"], axis=0))
            .to(self._device)
            .float()
        )
        encoded_legal_moves = action_encoding(
            observation["legal_moves_as_int"], self._act_dim
        )
        qvals = self._qnet(
            vectorized_observation, torch.tensor(encoded_legal_moves).to(self._device)
        ).cpu()

        if self._rng.random() < epsilon:
            action = np.random.choice(observation["legal_moves_as_int"]).item()
        else:
            ilegal_moves = [
                x
                for x in list(np.arange(qvals.shape[1]))
                if x not in observation["legal_moves_as_int"]
            ]
            qvals[:, ilegal_moves] = qvals.min()
            action = torch.argmax(qvals).item()

        if self._logger.should_log(self._timescale) and self._state["episode_start"]:
            self._logger.log_scalar(
                "train_qval" if self._training else "test_qval",
                torch.max(qvals),
                self._timescale,
            )
            self._state["episode_start"] = False

        return action

    def update(self, update_info):
        """
        Updates the DQN agent.

        Args:
            update_info: dictionary containing all the necessary information to
            update the agent. Should contain a full transition, with keys for
            "observation", "action", "reward", "next_observation", "done", "legal_moves", and "next_legal_moves".
        """
        if update_info["done"]:
            self._state["episode_start"] = True

        if self._reward_clip is not None:
            update_info["reward"] = np.clip(
                update_info["reward"], -self._reward_clip, self._reward_clip
            )

        # Add the most recent transition to the replay buffer.
        if self._training:
            self._replay_buffer.add(
                update_info["observation"]["vectorized"],
                update_info["action"],
                update_info["reward"],
                update_info["done"],
                legal_moves=action_encoding(
                    update_info["observation"]["legal_moves_as_int"], self._act_dim
                ),
            )

        # Update the q network based on a sample batch from the replay buffer.
        # If the replay buffer doesn't have enough samples, catch the exception
        # and move on.
        if (
            self._learn_schedule.update()
            and self._replay_buffer.size() > 0
            and self._update_period_schedule.update()
        ):
            batch = self._replay_buffer.sample(batch_size=self._batch_size)
            for key in batch:
                batch[key] = torch.tensor(batch[key]).to(self._device)

            # Compute predicted Q values
            self._optimizer.zero_grad()
            pred_qvals = self._qnet(batch["observation"], batch["legal_moves"])
            actions = batch["action"].long()

            if self._distributional:
                current_dist = self._qnet.dist(
                    batch["observation"], batch["legal_moves"]
                )
                log_p = torch.log(
                    current_dist[torch.arange(batch["observation"].size(0)), actions]
                )
                with torch.no_grad():
                    target_prob = self.target_projection(
                        batch["next_observation"],
                        batch["next_legal_moves"],
                        batch["reward"],
                        batch["done"],
                    )

                loss = -(target_prob * log_p).sum(-1)

            else:
                pred_qvals = pred_qvals[torch.arange(pred_qvals.size(0)), actions]

                # Compute 1-step Q targets
                if self._double:
                    next_action = self._qnet(
                        batch["next_observation"], batch["next_legal_moves"]
                    )
                else:
                    next_action = self._target_qnet(
                        batch["next_observation"], batch["next_legal_moves"]
                    )

                _, next_action = torch.max(next_action, dim=1)
                next_qvals = self._target_qnet(
                    batch["next_observation"], batch["next_legal_moves"]
                )
                next_qvals = next_qvals[torch.arange(next_qvals.size(0)), next_action]

                q_targets = batch["reward"] + self._discount_rate * next_qvals * (
                    1 - batch["done"]
                )

                loss = self._loss_fn(pred_qvals, q_targets)

            if isinstance(self._replay_buffer, PrioritizedReplayBuffer):
                td_errors = loss.sqrt().detach().cpu().numpy()
                self._replay_buffer.update_priorities(batch["indices"], td_errors)
                loss *= batch["weights"]
            loss = loss.mean()

            if self._logger.should_log(self._timescale):
                self._logger.log_scalar(
                    "train_loss" if self._training else "test_loss",
                    loss,
                    self._timescale,
                )
            if self._training:
                loss.backward()
                if self._grad_clip is not None:
                    torch.nn.utils.clip_grad_value_(
                        self._qnet.parameters(), self._grad_clip
                    )
                self._optimizer.step()

        # Update target network
        if self._training and self._target_net_update_schedule.update():
            self._update_target()


def action_encoding(legal_moves, act_dim):
    encoded_legal_moves = np.zeros(act_dim, dtype=np.int8)
    encoded_legal_moves[legal_moves] = 1
    return encoded_legal_moves
