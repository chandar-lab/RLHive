import copy
from functools import partial
from typing import Tuple

import numpy as np
import torch
from hive.agents.dqn import DQNAgent
from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.noisy_linear import NoisyLinear
from hive.agents.qnets.qnet_heads import (
    DistributionalNetwork,
    DQNNetwork,
    DuelingNetwork,
)
from hive.agents.qnets.utils import calculate_output_dim
from hive.replays import PrioritizedReplayBuffer
from hive.replays.replay_buffer import BaseReplayBuffer
from hive.utils.logging import Logger
from hive.utils.schedule import Schedule
from hive.utils.utils import OptimizerFn


class RainbowDQNAgent(DQNAgent):
    """An agent implementing the Rainbow algorithm."""

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
        test_epsilon: float = 0.001,
        learn_schedule: Schedule = None,
        seed: int = 42,
        batch_size: int = 32,
        device: str = "cpu",
        logger: Logger = None,
        log_frequency: int = 100,
        noisy: bool = True,
        std_init: float = 0.5,
        double: bool = True,
        dueling: bool = True,
        distributional: bool = True,
        use_eps_greedy: bool = False,
    ):
        """
        Args:
            qnet: A network that outputs the q-values of the different actions
                for an input observation.
            obs_dim: The dimension of the observations.
            act_dim: The number of actions available to the agent.
            v_min: minimum possible value of the value function
            v_max: maximum possible value of the value function
            atoms: number of atoms in the distributional DQN context
            optimizer_fn: A function that takes in a list of parameters to optimize
                and returns the optimizer.
            id: ID used to create the timescale in the logger for the agent.
            replay_buffer: The replay buffer that the agent will push observations
                to and sample from during learning.
            discount_rate (float): A number between 0 and 1 specifying how much
                future rewards are discounted by the agent.
            grad_clip (float): Gradients will be clipped to between
                [-grad_clip, gradclip]
            reward_clip (float): Rewards will be clipped to between
                [-reward_clip, reward_clip]
            target_net_soft_update (bool): Whether the target net parameters are
                replaced by the qnet parameters completely or using a weighted
                average of the target net parameters and the qnet parameters.
            target_net_update_fraction (float): The weight given to the target
                net parameters in a soft update.
            target_net_update_schedule: Schedule determining how frequently the
                target net is updated.
            update_period_schedule: Schedule determining how frequently
                the agent's net is updated.
            epsilon_schedule: Schedule determining the value of epsilon through
                the course of training.
            learn_schedule: Schedule determining when the learning process actually
                starts.
            seed: Seed for numpy random number generator.
            batch_size (int): The size of the batch sampled from the replay buffer
                during learning.
            device: Device on which all computations should be run.
            logger: Logger used to log agent's metrics.
            log_frequency (int): How often to log the agent's metrics.
            double: whether or not to use the double feature (from double DQN)
            distributional: whether or not to use the distributional
                feature (from distributional DQN)
            use_eps_greedy: whether or not to use epsilon greedy.
                Usually in case of noisy networks use_eps_greedy=False
        """
        self._noisy = noisy
        self._std_init = std_init
        self._double = double
        self._dueling = dueling
        self._distributional = distributional

        self._atoms = atoms if self._distributional else 1
        self._v_min = v_min
        self._v_max = v_max
        self._supports = torch.linspace(
            self._v_min, self._v_max, self._atoms, device=device
        )

        super().__init__(
            qnet,
            obs_dim,
            act_dim,
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
            test_epsilon=test_epsilon,
            learn_schedule=learn_schedule,
            seed=seed,
            batch_size=batch_size,
            device=device,
            logger=logger,
            log_frequency=log_frequency,
        )

        self._loss_fn = torch.nn.MSELoss(reduction="none")
        self._use_eps_greedy = use_eps_greedy

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
            network = DuelingNetwork(
                network, network_output_dim, self._act_dim, linear_fn, self._atoms
            )
        else:
            network = DQNNetwork(
                network, network_output_dim, self._act_dim * self._atoms, linear_fn
            )

        # Set up DistributionalNetwork wrapper if distributional is true
        if self._distributional:
            self._qnet = DistributionalNetwork(
                network, self._act_dim, self._v_min, self._v_max, self._atoms
            )
        else:
            self._qnet = network
        self._qnet.to(device=device)
        self._target_qnet = copy.deepcopy(self._qnet).requires_grad_(False)

    def target_projection(self, next_observation, reward, done):
        """Project distribution of target Q-values."""
        next_observation = next_observation.float()
        reward = reward.reshape(-1, 1)
        not_done = 1 - done.reshape(-1, 1)
        batch_size = next_observation.size(0)
        next_action = self._target_qnet(next_observation).argmax(1)
        next_dist = self._target_qnet.dist(next_observation)
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
            epsilon = self._test_epsilon

        observation = (
            torch.tensor(np.expand_dims(observation, axis=0)).to(self._device).float()
        )
        qvals = self._qnet(observation)

        if self._rng.random() < epsilon:
            action = self._rng.integers(self._act_dim)
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

    def update(self, update_info):
        """
        Updates the DQN agent.

        Args:
            update_info: dictionary containing all the necessary information to
            update the agent. Should contain a full transition, with keys for
            "observation", "action", "reward", "next_observation", and "done".
        """
        if update_info["done"]:
            self._state["episode_start"] = True

        if not self._training:
            return

        if self._reward_clip is not None:
            update_info["reward"] = np.clip(
                update_info["reward"], -self._reward_clip, self._reward_clip
            )

        # Add the most recent transition to the replay buffer.
        self._replay_buffer.add(
            update_info["observation"],
            update_info["action"],
            update_info["reward"],
            update_info["done"],
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
            pred_qvals = self._qnet(batch["observation"])
            actions = batch["action"].long()

            if self._distributional:
                current_dist = self._qnet.dist(batch["observation"])
                log_p = torch.log(
                    current_dist[torch.arange(batch["observation"].size(0)), actions]
                )
                with torch.no_grad():
                    target_prob = self.target_projection(
                        batch["next_observation"], batch["reward"], batch["done"]
                    )

                loss = -(target_prob * log_p).sum(-1)

            else:
                pred_qvals = pred_qvals[torch.arange(pred_qvals.size(0)), actions]

                # Compute 1-step Q targets
                if self._double:
                    next_action = self._qnet(batch["next_observation"])
                else:
                    next_action = self._target_qnet(batch["next_observation"])

                _, next_action = torch.max(next_action, dim=1)
                next_qvals = self._target_qnet(batch["next_observation"])
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
                    "train_loss",
                    loss,
                    self._timescale,
                )
            loss.backward()
            if self._grad_clip is not None:
                torch.nn.utils.clip_grad_value_(
                    self._qnet.parameters(), self._grad_clip
                )
            self._optimizer.step()

        # Update target network
        if self._target_net_update_schedule.update():
            self._update_target()
