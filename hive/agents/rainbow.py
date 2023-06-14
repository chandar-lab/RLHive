import copy
from functools import partial
from typing import Optional

import gymnasium as gym
import numpy as np
import torch

from hive.agents.dqn import DQNAgent
from hive.agents.networks.noisy_linear import NoisyLinear
from hive.agents.networks.qnet_heads import (
    DistributionalNetwork,
    DQNNetwork,
    DuelingNetwork,
)
from hive.agents.networks.utils import ModuleInitFn, calculate_output_dim
from hive.replays import PrioritizedReplayBuffer
from hive.replays.replay_buffer import BaseReplayBuffer
from hive.types import Creates, Partial, default
from hive.utils.loggers import logger
from hive.utils.schedule import Schedule
from hive.utils.utils import LossFn


class RainbowDQNAgent(DQNAgent):
    """An agent implementing the Rainbow algorithm."""

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
        replay_buffer: Optional[Creates[BaseReplayBuffer]] = None,
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
        """
        Args:
            observation_space (gym.spaces.Box): Observation space for the agent.
            action_space (gym.spaces.Discrete): Action space for the agent.
            representation_net (torch.nn.Module): A network that outputs the
                representations that will be used to compute Q-values (e.g.
                everything except the final layer of the DQN).
            stack_size: Number of observations stacked to create the state fed to the
                DQN.
            id: Agent identifier.
            optimizer_fn (torch.optim.Optimizer): A function that takes in a list of parameters
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
        loss_fn = default(loss_fn, torch.nn.MSELoss)
        replay_buffer = default(replay_buffer, PrioritizedReplayBuffer)

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            representation_net=representation_net,
            stack_size=stack_size,
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
            log_frequency=log_frequency,
        )

        self._supports = torch.linspace(
            self._v_min, self._v_max, self._atoms, device=self._device
        )

        self._use_eps_greedy = use_eps_greedy

    def create_networks(self, representation_net):
        """Creates the Q-network and target Q-network. Adds the appropriate heads
        for DQN, Dueling DQN, Noisy Networks, and Distributional DQN.

        Args:
            representation_net: A network that outputs the representations that will
                be used to compute Q-values (e.g. everything except the final layer
                of the DQN).
        """
        network = representation_net(self._state_size)
        network_output_dim = np.prod(calculate_output_dim(network, self._state_size))  # type: ignore

        # Use NoisyLinear when creating output heads if noisy is true
        linear_fn = (
            partial(NoisyLinear, std_init=self._std_init)
            if self._noisy
            else partial(torch.nn.Linear)
        )

        # Set up Dueling heads
        if self._dueling:
            network = DuelingNetwork(
                network,
                network_output_dim,
                int(self._action_space.n),
                linear_fn,
                self._atoms,
            )
        else:
            network = DQNNetwork(
                network,
                network_output_dim,
                int(self._action_space.n * self._atoms),
                linear_fn,
            )

        # Set up DistributionalNetwork wrapper if distributional is true
        if self._distributional:
            self._qnet = DistributionalNetwork(
                network,
                int(self._action_space.n),
                self._v_min,
                self._v_max,
                self._atoms,
            )
        else:
            self._qnet = network
        self._qnet.to(device=self._device)
        self._qnet.apply(self._init_fn)
        self._target_qnet = copy.deepcopy(self._qnet).requires_grad_(False)

    @torch.no_grad()
    def act(self, observation, agent_traj_state, global_step):
        """Returns the action for the agent. If in training mode, follows an epsilon
        greedy policy. Otherwise, returns the action with the highest Q-value.

        Args:
            observation: The current observation.
            agent_traj_state: Contains necessary state information for the agent
                to process current trajectory. This should be updated and returned.

        Returns:
            - action
            - agent trajectory state
        """

        # Determine and log the value of epsilon
        if self._training:
            if not self._learn_schedule(global_step):
                epsilon = 1.0
            else:
                epsilon = self._epsilon_schedule(global_step)
            if self._log_schedule(global_step):
                logger.log_scalar("epsilon", epsilon, self.id)
        else:
            epsilon = self._test_epsilon

        state, observation_stack = self.preprocess_observation(
            observation, agent_traj_state
        )

        # Sample action. With epsilon probability choose random action,
        # otherwise select the action with the highest q-value.
        qvals = self._qnet(*state)
        if self._rng.random() < epsilon:
            action = self._rng.integers(self._action_space.n)
        else:
            # Note: not explicitly handling the ties
            action = torch.argmax(qvals).item()

        if (
            self._training
            and self._log_schedule(global_step)
            and agent_traj_state is None
        ):
            logger.log_scalar("train_qval", torch.max(qvals), self.id)
        return action, {"observation_stack": observation_stack}

    def update(self, update_info, agent_traj_state, global_step):
        """
        Updates the DQN agent.

        Args:
            update_info: dictionary containing all the necessary information
                from the environment to update the agent. Should contain a full
                transition, with keys for "observation", "action", "reward",
                "next_observation", "terminated", and "truncated".
            agent_traj_state: Contains necessary state information for the agent
                to process current trajectory. This should be updated and returned.

        Returns:
            - action
            - agent trajectory state
        """
        if not self._training:
            return agent_traj_state

        # Add the most recent transition to the replay buffer.
        self._replay_buffer.add(**self.preprocess_update_info(update_info))

        # Update the q network based on a sample batch from the replay buffer.
        # If the replay buffer doesn't have enough samples, catch the exception
        # and move on.
        if (
            self._learn_schedule(global_step)
            and self._replay_buffer.size() > 0
            and self._update_period_schedule(global_step)
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
                current_dist = self._qnet.dist(
                    *current_state_inputs  # pyright: ignore reportGeneralTypeIssues
                )
                probs = current_dist[torch.arange(actions.size(0)), actions]
                probs = torch.clamp(probs, 1e-6, 1)  # NaN-guard
                log_p = torch.log(probs)
                with torch.no_grad():
                    target_prob = self.target_projection(
                        next_state_inputs,
                        next_action,
                        batch["reward"],
                        batch["terminated"],
                    )

                loss = -(target_prob * log_p).sum(-1)

            else:
                pred_qvals = pred_qvals[torch.arange(pred_qvals.size(0)), actions]

                next_qvals = self._target_qnet(*next_state_inputs)
                next_qvals = next_qvals[torch.arange(next_qvals.size(0)), next_action]

                q_targets = batch["reward"] + self._discount_rate * next_qvals * (
                    1 - batch["terminated"]
                )

                loss = self._loss_fn(pred_qvals, q_targets)

            if isinstance(self._replay_buffer, PrioritizedReplayBuffer):
                td_errors = loss.detach().cpu().numpy()
                self._replay_buffer.update_priorities(batch["indices"], td_errors)
                loss *= batch["weights"]
            loss = loss.mean()

            if self._log_schedule(global_step):
                logger.log_scalar(
                    "train_loss",
                    loss,
                    self.id,
                )
            loss.backward()
            if self._grad_clip is not None:
                torch.nn.utils.clip_grad_value_(  # type: ignore
                    self._qnet.parameters(), self._grad_clip
                )
            self._optimizer.step()

        # Update target network
        if self._target_net_update_schedule(global_step):
            self._update_target()
        return agent_traj_state

    def target_projection(self, target_net_inputs, next_action, reward, terminated):
        """Project distribution of target Q-values.

        Args:
            target_net_inputs: Inputs to feed into the target net to compute the
                projection of the target Q-values. Should be set from
                :py:meth:`~hive.agents.dqn.DQNAgent.preprocess_update_batch`.
            next_action (~torch.Tensor): Tensor containing next actions used to
                compute target distribution.
            reward (~torch.Tensor): Tensor containing rewards for the current batch.
            terminated (~torch.Tensor): Tensor containing whether the states in
            the current batch are terminal.

        """
        reward = reward.reshape(-1, 1)
        not_terminated = 1 - terminated.reshape(-1, 1)
        batch_size = reward.size(0)
        next_dist = self._target_qnet.dist(
            *target_net_inputs
        )  # pyright: ignore reportGeneralTypeIssues
        next_dist = next_dist[torch.arange(batch_size), next_action]

        dist_supports = reward + not_terminated * self._discount_rate * self._supports
        dist_supports = dist_supports.clamp(min=self._v_min, max=self._v_max)
        dist_supports = dist_supports.unsqueeze(1)
        dist_supports = dist_supports.tile([1, self._atoms, 1])
        projected_supports = self._supports.tile([batch_size, 1]).unsqueeze(2)

        delta = float(self._v_max - self._v_min) / (self._atoms - 1)
        quotient = 1 - (torch.abs(dist_supports - projected_supports) / delta)
        quotient = quotient.clamp(min=0, max=1)

        projection = torch.sum(quotient * next_dist.unsqueeze(1), dim=2)
        return projection
