import copy
import os
from collections import deque

import gymnasium as gym
from gymnasium.vector.utils.numpy_utils import create_empty_array, concatenate
import numpy as np
import torch
from gymnasium.vector.utils.numpy_utils import create_empty_array

from hive.agents.agent import Agent
from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.qnet_heads import DQNNetwork
from hive.agents.qnets.utils import (
    InitializationFn,
    calculate_output_dim,
    create_init_weights_fn,
)
from hive.agents.utils import roll_state
from hive.replays import BaseReplayBuffer, CircularReplayBuffer, ReplayItemSpec
from hive.utils.loggers import logger
from hive.utils.schedule import (
    LinearSchedule,
    PeriodicSchedule,
    Schedule,
    SwitchSchedule,
)
from hive.utils.utils import LossFn, create_folder, seeder, DL_to_LD
from hive.utils.registry import Creates, OCreates, default
from typing import Optional
from functools import partial


class DQNAgent(Agent[gym.spaces.Box, gym.spaces.Discrete]):
    """An agent implementing the DQN algorithm. Uses an epsilon greedy
    exploration policy
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        representation_net: Creates[FunctionApproximator],
        stack_size: int = 1,
        id=0,
        optimizer_fn: OCreates[torch.optim.Optimizer] = None,
        loss_fn: OCreates[LossFn] = None,
        init_fn: OCreates[InitializationFn] = None,
        replay_buffer: OCreates[BaseReplayBuffer] = None,
        discount_rate: float = 0.99,
        n_step: int = 1,
        grad_clip: Optional[float] = None,
        reward_clip: Optional[float] = None,
        update_period_schedule: OCreates[Schedule[bool]] = None,
        target_net_soft_update: bool = False,
        target_net_update_fraction: float = 0.05,
        target_net_update_schedule: OCreates[Schedule[bool]] = None,
        epsilon_schedule: OCreates[Schedule[float]] = None,
        test_epsilon: float = 0.001,
        min_replay_history: int = 5000,
        batch_size: int = 32,
        device="cpu",
        log_frequency: int = 100,
    ):
        """
        Args:
            observation_space (gym.spaces.Box): Observation space for the agent.
            action_space (gym.spaces.Discrete): Action space for the agent.
            representation_net (FunctionApproximator): A network that outputs the
                representations that will be used to compute Q-values (e.g.
                everything except the final layer of the DQN).
            stack_size: Number of observations stacked to create the state fed to the
                DQN.
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
                :py:class:`~hive.replays.circular_replay.CircularReplayBuffer`.
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
        """
        super().__init__(
            observation_space=observation_space, action_space=action_space, id=id
        )
        self._stack_size = stack_size
        self._state_size = (
            stack_size * self._observation_space.shape[0],
            *self._observation_space.shape[1:],
        )
        self._init_fn = create_init_weights_fn(init_fn)
        self._device = torch.device("cpu" if not torch.cuda.is_available() else device)
        self.create_q_networks(representation_net)
        optimizer_fn = default(optimizer_fn, torch.optim.Adam)
        self._optimizer = optimizer_fn(self._qnet.parameters())
        self._rng = np.random.default_rng(seed=seeder.get_new_seed("agent"))
        replay_buffer = default(replay_buffer, CircularReplayBuffer)
        self._replay_buffer = replay_buffer(
            observation_spec=ReplayItemSpec.create(
                shape=self._observation_space.shape, dtype=self._observation_space.dtype
            ),
            action_spec=ReplayItemSpec.create(
                shape=self._action_space.shape, dtype=self._action_space.dtype
            ),
            stack_size=stack_size,
            gamma=discount_rate,
            n_step=n_step,
        )
        self._discount_rate = discount_rate**n_step
        self._grad_clip = grad_clip
        self._reward_clip = reward_clip
        self._target_net_soft_update = target_net_soft_update
        self._target_net_update_fraction = target_net_update_fraction
        loss_fn = default(loss_fn, torch.nn.SmoothL1Loss)
        self._loss_fn = loss_fn(reduction="none")
        self._batch_size = batch_size
        self._log_schedule = PeriodicSchedule(False, True, log_frequency)
        update_period_schedule = default(
            update_period_schedule, lambda: PeriodicSchedule(False, True, 1)
        )
        self._update_period_schedule = update_period_schedule()

        target_net_update_schedule = default(
            target_net_update_schedule, lambda: PeriodicSchedule(False, True, 10000)
        )
        self._target_net_update_schedule = target_net_update_schedule()

        epsilon_schedule = default(
            epsilon_schedule, lambda: LinearSchedule(1, 0.1, 100000)
        )
        self._epsilon_schedule = epsilon_schedule()
        self._test_epsilon = test_epsilon
        self._learn_schedule = SwitchSchedule(False, True, min_replay_history)

        self._training = False

    def create_q_networks(self, representation_net):
        """Creates the Q-network and target Q-network.

        Args:
            representation_net: A network that outputs the representations that will
                be used to compute Q-values (e.g. everything except the final layer
                of the DQN).
        """
        network = representation_net(self._state_size)
        network_output_dim = np.prod(calculate_output_dim(network, self._state_size))  # type: ignore
        self._qnet = DQNNetwork(
            network, network_output_dim, int(self._action_space.n)
        ).to(self._device)
        self._qnet.apply(self._init_fn)
        self._target_qnet = copy.deepcopy(self._qnet).requires_grad_(False)

    def train(self):
        """Changes the agent to training mode."""
        super().train()
        self._qnet.train()
        self._target_qnet.train()

    def eval(self):
        """Changes the agent to evaluation mode."""
        super().eval()
        self._qnet.eval()
        self._target_qnet.eval()

    def preprocess_observation(self, observation, agent_traj_state):
        if agent_traj_state is None:
            observation_stack = create_empty_array(
                self._observation_space, n=self._stack_size
            )
        else:
            observation_stack = agent_traj_state["observation_stack"]
        observation_stack = roll_state(observation, observation_stack)
        state = (
            torch.tensor(observation_stack, device=self._device, dtype=torch.float32)
            .flatten(0, 1)
            .unsqueeze(0)
        )
        return state, observation_stack

    def preprocess_update_info(self, update_info):
        """Preprocesses the :obj:`update_info` before it goes into the replay buffer.
        Clips the reward in update_info.

        Args:
            update_info: Contains the information from the current timestep that the
                agent should use to update itself.
        """
        if self._reward_clip is not None:
            update_info["reward"] = np.clip(
                update_info["reward"], -self._reward_clip, self._reward_clip
            )
        preprocessed_update_info = {
            "observation": update_info["observation"],
            "next_observation": update_info["next_observation"],
            "action": update_info["action"],
            "reward": update_info["reward"],
            "terminated": update_info["terminated"],
            "truncated": update_info["truncated"],
            "source": update_info["source"],
        }

        return preprocessed_update_info

    def preprocess_update_batch(self, batch):
        """Preprocess the batch sampled from the replay buffer.

        Args:
            batch: Batch sampled from the replay buffer for the current update.

        Returns:
            (tuple):
                - (tuple) Inputs used to calculate current state values.
                - (tuple) Inputs used to calculate next state values
                - Preprocessed batch.
        """
        for key in batch:
            batch[key] = torch.tensor(batch[key], device=self._device)
        return (batch["observation"],), (batch["next_observation"],), batch

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
        qvals = self._qnet(state)
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
        agent_traj_state = {"observation_stack": observation_stack}
        return action, agent_traj_state

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
            return

        # Add the most recent transition to the replay buffer.
        transition = self.preprocess_update_info(update_info)
        self._replay_buffer.add(**transition)

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
            pred_qvals = self._qnet(current_state_inputs)
            actions = batch["action"].long()
            pred_qvals = pred_qvals[torch.arange(pred_qvals.size(0)), actions]

            # Compute 1-step Q targets
            next_qvals = self._target_qnet(next_state_inputs)
            next_qvals, _ = torch.max(next_qvals, dim=1)

            q_targets = batch["reward"] + self._discount_rate * next_qvals * (
                1 - batch["terminated"]
            )

            loss = self._loss_fn(pred_qvals, q_targets).mean()

            if self._log_schedule(global_step):
                logger.log_scalar("train_loss", loss, self.id)

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

    def _update_target(self):
        """Update the target network."""
        if self._target_net_soft_update:
            target_params = self._target_qnet.state_dict()
            current_params = self._qnet.state_dict()
            for key in list(target_params.keys()):
                target_params[key] = (
                    1 - self._target_net_update_fraction
                ) * target_params[
                    key
                ] + self._target_net_update_fraction * current_params[
                    key
                ]
            self._target_qnet.load_state_dict(target_params)
        else:
            self._target_qnet.load_state_dict(self._qnet.state_dict())

    def save(self, dname):
        torch.save(
            {
                "qnet": self._qnet.state_dict(),
                "target_qnet": self._target_qnet.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "rng": self._rng,
            },
            os.path.join(dname, "agent.pt"),
        )
        replay_dir = os.path.join(dname, "replay")
        create_folder(replay_dir)
        self._replay_buffer.save(replay_dir)

    def load(self, dname):
        checkpoint = torch.load(os.path.join(dname, "agent.pt"))
        self._qnet.load_state_dict(checkpoint["qnet"])
        self._target_qnet.load_state_dict(checkpoint["target_qnet"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])
        self._rng = checkpoint["rng"]
        self._replay_buffer.load(os.path.join(dname, "replay"))
