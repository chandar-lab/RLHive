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
from hive.agents.qnets.utils import InitializationFn, calculate_output_dim
from hive.replays import PrioritizedReplayBuffer
from hive.replays.replay_buffer import BaseReplayBuffer
from hive.utils.loggers import Logger
from hive.utils.schedule import Schedule
from hive.utils.utils import LossFn, OptimizerFn, seeder


class RainbowDQNAgent(DQNAgent):
    """An agent implementing the Rainbow algorithm."""

    def __init__(
        self,
        representation_net: FunctionApproximator,
        obs_dim: Tuple,
        act_dim: int,
        optimizer_fn: OptimizerFn = None,
        loss_fn: LossFn = None,
        init_fn: InitializationFn = None,
        id=0,
        replay_buffer: BaseReplayBuffer = None,
        discount_rate: float = 0.99,
        n_step: int = 1,
        grad_clip: float = None,
        reward_clip: float = None,
        update_period_schedule: Schedule = None,
        target_net_soft_update: bool = False,
        target_net_update_fraction: float = 0.05,
        target_net_update_schedule: Schedule = None,
        epsilon_schedule: Schedule = None,
        test_epsilon: float = 0.001,
        min_replay_history: int = 5000,
        batch_size: int = 32,
        device="cpu",
        logger: Logger = None,
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
        """
        Args:
            representation_net (FunctionApproximator): A network that outputs the
                representations that will be used to compute Q-values (e.g.
                everything except the final layer of the DQN).
            obs_dim: The shape of the observations.
            act_dim (int): The number of actions available to the agent.
            id: Agent identifier.
            optimizer_fn (OptimizerFn): A function that takes in a list of parameters
                to optimize and returns the optimizer. If None, defaults to
                :py:class:`~torch.optim.Adam`.
            loss_fn (LossFn): Loss function used by the agent. If None, defaults to
                :py:class:`~torch.nn.SmoothL1Loss`.
            init_fn (InitializationFn): Initializes the weights of qnet using
                create_init_weights_fn.
            replay_buffer (BaseReplayBuffer): The replay buffer that the agent will
                push observations to and sample from during learning. If None,
                defaults to
                :py:class:`~hive.replays.prioritized_replay.PrioritizedReplayBuffer`.
            discount_rate (float): A number between 0 and 1 specifying how much
                future rewards are discounted by the agent.
            n_step (int): The horizon used in n-step returns to compute TD(n) targets.
            grad_clip (float): Gradients will be clipped to between
                [-grad_clip, grad_clip].
            reward_clip (float): Rewards will be clipped to between
                [-reward_clip, reward_clip].
            update_period_schedule (Schedule): Schedule determining how frequently
                the agent's Q-network is updated.
            target_net_soft_update (bool): Whether the target net parameters are
                replaced by the qnet parameters completely or using a weighted
                average of the target net parameters and the qnet parameters.
            target_net_update_fraction (float): The weight given to the target
                net parameters in a soft update.
            target_net_update_schedule (Schedule): Schedule determining how frequently
                the target net is updated.
            epsilon_schedule (Schedule): Schedule determining the value of epsilon
                through the course of training.
            test_epsilon (float): epsilon (probability of choosing a random action)
                to be used during testing phase.
            min_replay_history (int): How many observations to fill the replay buffer
                with before starting to learn.
            batch_size (int): The size of the batch sampled from the replay buffer
                during learning.
            device: Device on which all computations should be run.
            logger (ScheduledLogger): Logger used to log agent's metrics.
            log_frequency (int): How often to log the agent's metrics.
            noisy (bool): Whether to use noisy linear layers for exploration.
            std_init (float): The range for the initialization of the standard
                deviation of the weights.
            use_eps_greedy (bool): Whether to use epsilon greedy exploration.
            double (bool): Whether to use double DQN.
            dueling (bool): Whether to use a dueling network architecture.
            distributional (bool): Whether to use the distributional RL.
            vmin (float): The minimum of the support of the categorical value
                distribution for distributional RL.
            vmax (float): The maximum of the support of the categorical value
                distribution for distributional RL.
            atoms (int): Number of atoms discretizing the support range of the
                categorical value distribution for distributional RL.
        """

        self._noisy = noisy
        self._std_init = std_init
        self._double = double
        self._dueling = dueling
        self._distributional = distributional

        self._atoms = atoms if self._distributional else 1
        self._v_min = v_min
        self._v_max = v_max

        if loss_fn is None:
            loss_fn = torch.nn.MSELoss

        if replay_buffer is None:
            replay_buffer = PrioritizedReplayBuffer(seed=seeder.get_new_seed())

        super().__init__(
            representation_net,
            obs_dim,
            act_dim,
            optimizer_fn=optimizer_fn,
            init_fn=init_fn,
            loss_fn=loss_fn,
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
            min_replay_history=min_replay_history,
            batch_size=batch_size,
            device=device,
            logger=logger,
            log_frequency=log_frequency,
        )

        self._supports = torch.linspace(
            self._v_min, self._v_max, self._atoms, device=self._device
        )

        self._use_eps_greedy = use_eps_greedy

    def create_q_networks(self, representation_net):
        """Creates the Q-network and target Q-network. Adds the appropriate heads
        for DQN, Dueling DQN, Noisy Networks, and Distributional DQN.

        Args:
            representation_net: A network that outputs the representations that will
                be used to compute Q-values (e.g. everything except the final layer
                of the DQN).
        """
        network = representation_net(self._obs_dim)
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
        self._qnet.to(device=self._device)
        self._qnet.apply(self._init_fn)
        self._target_qnet = copy.deepcopy(self._qnet).requires_grad_(False)

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

        observation = torch.tensor(
            np.expand_dims(observation, axis=0), device=self._device
        ).float()
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

        # Add the most recent transition to the replay buffer.
        self._replay_buffer.add(**self.preprocess_update_info(update_info))

        # Update the q network based on a sample batch from the replay buffer.
        # If the replay buffer doesn't have enough samples, catch the exception
        # and move on.
        if (
            self._learn_schedule.update()
            and self._replay_buffer.size() > 0
            and self._update_period_schedule.update()
        ):
            batch = self._replay_buffer.sample(batch_size=self._batch_size)
            (
                current_state_inputs,
                next_state_inputs,
                batch,
            ) = self.preprocess_update_batch(batch)

            # Compute predicted Q values
            self._optimizer.zero_grad()
            pred_qvals = self._qnet(*current_state_inputs)
            actions = batch["action"].long()

            if self._double:
                next_action = self._qnet(*next_state_inputs)
            else:
                next_action = self._target_qnet(*next_state_inputs)
            next_action = next_action.argmax(1)

            if self._distributional:
                current_dist = self._qnet.dist(*current_state_inputs)
                probs = current_dist[torch.arange(actions.size(0)), actions]
                probs = torch.clamp(probs, 1e-6, 1)  # NaN-guard
                log_p = torch.log(probs)
                with torch.no_grad():
                    target_prob = self.target_projection(
                        next_state_inputs, next_action, batch["reward"], batch["done"]
                    )

                loss = -(target_prob * log_p).sum(-1)

            else:
                pred_qvals = pred_qvals[torch.arange(pred_qvals.size(0)), actions]

                next_qvals = self._target_qnet(*next_state_inputs)
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

    def target_projection(self, target_net_inputs, next_action, reward, done):
        """Project distribution of target Q-values.

        Args:
            target_net_inputs: Inputs to feed into the target net to compute the
                projection of the target Q-values. Should be set from
                :py:meth:`~hive.agents.dqn.DQNAgent.preprocess_update_batch`.
            next_action (~torch.Tensor): Tensor containing next actions used to
                compute target distribution.
            reward (~torch.Tensor): Tensor containing rewards for the current batch.
            done (~torch.Tensor): Tensor containing whether the states in the current
                batch are terminal.

        """
        reward = reward.reshape(-1, 1)
        not_done = 1 - done.reshape(-1, 1)
        batch_size = reward.size(0)
        next_dist = self._target_qnet.dist(*target_net_inputs)
        next_dist = next_dist[torch.arange(batch_size), next_action]

        dist_supports = reward + not_done * self._discount_rate * self._supports
        dist_supports = dist_supports.clamp(min=self._v_min, max=self._v_max)
        dist_supports = dist_supports.unsqueeze(1)
        dist_supports = dist_supports.tile([1, self._atoms, 1])
        projected_supports = self._supports.tile([batch_size, 1]).unsqueeze(2)

        delta = float(self._v_max - self._v_min) / (self._atoms - 1)
        quotient = 1 - (torch.abs(dist_supports - projected_supports) / delta)
        quotient = quotient.clamp(min=0, max=1)

        projection = torch.sum(quotient * next_dist.unsqueeze(1), dim=2)
        return projection
