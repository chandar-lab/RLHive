import copy
import os
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector.utils.numpy_utils import create_empty_array

from hive.agents.agent import Agent
from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.td3_heads import TD3ActorNetwork, TD3CriticNetwork
from hive.agents.qnets.utils import (
    InitializationFn,
    calculate_output_dim,
    create_init_weights_fn,
)
from hive.agents.utils import roll_state
from hive.replays import BaseReplayBuffer, CircularReplayBuffer, ReplayItemSpec
from hive.utils.loggers import logger
from hive.utils.schedule import PeriodicSchedule, SwitchSchedule
from hive.utils.utils import LossFn, OptimizerFn, create_folder

from hive.utils.registry import OCreates, default
from typing import Optional, cast
from hive.types import Shape


class TD3(Agent):
    """An agent implementing the TD3 algorithm."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        representation_net: OCreates[FunctionApproximator] = None,
        actor_net: OCreates[FunctionApproximator] = None,
        critic_net: OCreates[FunctionApproximator] = None,
        init_fn: OCreates[InitializationFn] = None,
        actor_optimizer_fn: OCreates[OptimizerFn] = None,
        critic_optimizer_fn: OCreates[OptimizerFn] = None,
        critic_loss_fn: OCreates[LossFn] = None,
        n_critics: int = 2,
        stack_size: int = 1,
        replay_buffer: OCreates[BaseReplayBuffer] = None,
        discount_rate: float = 0.99,
        n_step: int = 1,
        grad_clip: Optional[float] = None,
        reward_clip: Optional[float] = None,
        soft_update_fraction: float = 0.005,
        batch_size: int = 64,
        log_frequency: int = 100,
        update_frequency: int = 1,
        policy_update_frequency: int = 2,
        action_noise: float = 0,
        target_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        min_replay_history: int = 1000,
        device="cpu",
        id=0,
    ):
        """
        Args:
            observation_space (gym.spaces.Box): Observation space for the agent.
            action_space (gym.spaces.Box): Action space for the agent.
            representation_net (FunctionApproximator): The network that encodes the
                observations that are then fed into the actor_net and critic_net. If
                None, defaults to :py:class:`~torch.nn.Identity`.
            actor_net (FunctionApproximator): The network that takes the encoded
                observations from representation_net and outputs the representations
                used to compute the actions (ie everything except the last layer).
            critic_net (FunctionApproximator): The network that takes two inputs: the
                encoded observations from representation_net and actions. It outputs
                the representations used to compute the values of the actions (ie
                everything except the last layer).
            init_fn (InitializationFn): Initializes the weights of agent networks using
                create_init_weights_fn.
            actor_optimizer_fn (OptimizerFn): A function that takes in the list of
                parameters of the actor returns the optimizer for the actor. If None,
                defaults to :py:class:`~torch.optim.Adam`.
            critic_optimizer_fn (OptimizerFn): A function that takes in the list of
                parameters of the critic returns the optimizer for the critic. If None,
                defaults to :py:class:`~torch.optim.Adam`.
            critic_loss_fn (LossFn): The loss function used to optimize the critic. If
                None, defaults to :py:class:`~torch.nn.MSELoss`.
            n_critics (int): The number of critics used by the agent to estimate
                Q-values. The minimum Q-value is used as the value for the next state
                when calculating target Q-values for the critic. The output of the
                first critic is used when computing the loss for the actor. For TD3,
                the default value is 2. For DDPG, this parameter is 1.
            stack_size (int): Number of observations stacked to create the state fed
                to the agent.
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
            soft_update_fraction (float): The weight given to the target
                net parameters in a soft (polyak) update. Also known as tau.
            batch_size (int): The size of the batch sampled from the replay buffer
                during learning.
            logger (Logger): Logger used to log agent's metrics.
            log_frequency (int): How often to log the agent's metrics.
            update_frequency (int): How frequently to update the agent. A value of 1
                means the agent will be updated every time update is called.
            policy_update_frequency (int): Relative update frequency of the actor
                compared to the critic. The actor will be updated every
                policy_update_frequency times the critic is updated.
            action_noise (float): The standard deviation for the noise added to the
                action taken by the agent during training.
            target_noise (float): The standard deviation of the noise added to the
                target policy for smoothing.
            target_noise_clip (float): The sampled target_noise is clipped to
                [-target_noise_clip, target_noise_clip].
            min_replay_history (int): How many observations to fill the replay buffer
                with before starting to learn.
            device: Device on which all computations should be run.
            id: Agent identifier.
        """
        super().__init__(observation_space, action_space, id)
        self._device = torch.device("cpu" if not torch.cuda.is_available() else device)
        self._stack_size = stack_size
        self._state_size = (
            stack_size * self._observation_space.shape[0],
            *self._observation_space.shape[1:],
        )
        self._action_min = self._action_space.low
        self._action_max = self._action_space.high
        self._action_scaling = 0.5 * (self._action_max - self._action_min)
        self._scale_actions = np.isfinite(self._action_scaling).all()
        self._action_min_tensor = torch.as_tensor(self._action_min, device=self._device)
        self._action_max_tensor = torch.as_tensor(self._action_max, device=self._device)
        self._init_fn = create_init_weights_fn(init_fn)
        self._n_critics = n_critics
        self.create_networks(representation_net, actor_net, critic_net)
        critic_optimizer_fn = default(critic_optimizer_fn, torch.optim.Adam)
        actor_optimizer_fn = default(actor_optimizer_fn, torch.optim.Adam)
        self._critic_optimizer = critic_optimizer_fn(self._critic.parameters())
        self._actor_optimizer = actor_optimizer_fn(self._actor.parameters())
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
        )
        self._discount_rate = discount_rate**n_step
        self._grad_clip = grad_clip
        self._reward_clip = reward_clip
        self._soft_update_fraction = soft_update_fraction
        critic_loss_fn = default(critic_loss_fn, torch.nn.MSELoss)
        self._critic_loss_fn = critic_loss_fn(reduction="mean")
        self._batch_size = batch_size
        self._log_schedule = PeriodicSchedule(False, True, log_frequency)
        self._update_schedule = PeriodicSchedule(False, True, update_frequency)
        self._policy_update_schedule = PeriodicSchedule(
            False, True, policy_update_frequency * update_frequency
        )
        self._action_noise = action_noise
        self._target_noise = target_noise
        self._target_noise_clip = target_noise_clip
        self._learn_schedule = SwitchSchedule(False, True, min_replay_history)

        self._training = False

    def create_networks(self, representation_net, actor_net, critic_net):
        """Creates the actor and critic networks.

        Args:
            representation_net: A network that outputs the shared representations that
                will be used by the actor and critic networks to process observations.
            actor_net: The network that will be used to compute actions.
            critic_net: The network that will be used to compute values of state action
                pairs.
        """
        representation_net = default(representation_net, torch.nn.Identity)
        network = representation_net(self._state_size)
        network_output_shape = cast(
            Shape, calculate_output_dim(network, self._state_size)
        )
        self._actor = TD3ActorNetwork(
            network,
            actor_net,
            network_output_shape,
            self._action_space.shape,
            self._scale_actions,
        ).to(self._device)
        self._critic = TD3CriticNetwork(
            network,
            network_output_shape,
            self._n_critics,
            self._action_space.shape,
            critic_net,
        ).to(self._device)

        self._actor.apply(self._init_fn)
        self._critic.apply(self._init_fn)
        self._target_actor = copy.deepcopy(self._actor).requires_grad_(False)
        self._target_critic = copy.deepcopy(self._critic).requires_grad_(False)

    def train(self):
        """Changes the agent to training mode."""
        super().train()
        self._actor.train()
        self._critic.train()
        self._target_actor.train()
        self._target_critic.train()

    def eval(self):
        """Changes the agent to evaluation mode."""
        super().eval()
        self._actor.eval()
        self._critic.eval()
        self._target_actor.eval()
        self._target_critic.eval()

    def scale_action(self, actions):
        """Scales actions to [-1, 1]."""
        if self._scale_actions:
            return ((actions - self._action_min) / self._action_scaling) - 1.0
        else:
            return actions

    def unscale_actions(self, actions):
        """Unscales actions from [-1, 1] to expected scale."""
        if self._scale_actions:
            return ((actions + 1.0) * self._action_scaling) + self._action_min
        else:
            return actions

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
        Scales the action to [-1, 1].

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
            "action": self.scale_action(update_info["action"]),
            "reward": update_info["reward"],
            "terminated": update_info["terminated"],
            "truncated": update_info["truncated"],
        }
        if "agent_id" in update_info:
            preprocessed_update_info["agent_id"] = int(update_info["agent_id"])

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
        """Returns the action for the agent. If in training mode, adds noise with
        standard deviation :py:obj:`self._action_noise`.

        Args:
            observation: The current observation.
            agent_traj_state: Contains necessary state information for the agent
                to process current trajectory. This should be updated and returned.

        Returns:
            - action
            - agent trajectory state
        """
        # Calculate action and add noise if training.
        state, observation_stack = self.preprocess_observation(
            observation, agent_traj_state
        )
        if self._training and not self._learn_schedule(global_step):
            return (
                self._action_space.sample(),
                {"observation_stack": observation_stack},
            )
        action = self._actor(state)
        if self._training:
            noise = torch.randn_like(action, requires_grad=False) * self._action_noise
            action = action + noise
        action = action.cpu().detach().numpy()
        if self._scale_actions:
            action = self.unscale_actions(action)
        action = np.clip(action, self._action_min, self._action_max)
        return np.squeeze(action, axis=0), {"observation_stack": observation_stack}

    def update(self, update_info, agent_traj_state, global_step):
        """
        Updates the TD3 agent.

        Args:
            update_info: dictionary containing all the necessary information
                from the environment to update the agent. Should contain a full
                transition, with keys for "observation", "action", "reward",
                "next_observation", "terminated", and "truncated
            agent_traj_state: Contains necessary state information for the agent
                to process current trajectory. This should be updated and returned.

        Returns:
            - action
            - agent trajectory state
        """

        if not self._training:
            return

        # Add the most recent transition to the replay buffer.
        self._replay_buffer.add(**self.preprocess_update_info(update_info))

        # Update the agent based on a sample batch from the replay buffer.
        if (
            self._learn_schedule(global_step)
            and self._replay_buffer.size() > 0
            and self._update_schedule(global_step)
        ):
            batch = self._replay_buffer.sample(batch_size=self._batch_size)
            (
                current_state_inputs,
                next_state_inputs,
                batch,
            ) = self.preprocess_update_batch(batch)
            with torch.no_grad():
                noise = torch.randn_like(batch["action"]) * self._target_noise
                noise = torch.clamp(
                    noise, -self._target_noise_clip, self._target_noise_clip
                )
                next_actions = self._target_actor(next_state_inputs) + noise
                if self._scale_actions:
                    next_actions = torch.clamp(next_actions, -1, 1)
                else:
                    next_actions = torch.clamp(
                        next_actions, self._action_min_tensor, self._action_max_tensor
                    )

                next_q_vals = torch.cat(
                    self._target_critic(next_state_inputs, next_actions), dim=1
                )
                next_q_vals, _ = torch.min(next_q_vals, dim=1, keepdim=True)
                target_q_values = (
                    batch["reward"][:, None]
                    + (1 - batch["terminated"][:, None])
                    * self._discount_rate
                    * next_q_vals
                )

            # Critic losses
            pred_qvals = self._critic(current_state_inputs, batch["action"])
            critic_loss = torch.stack(
                [self._critic_loss_fn(qvals, target_q_values) for qvals in pred_qvals]
            ).sum()
            self._critic_optimizer.zero_grad()
            critic_loss.backward()
            if self._grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(  # pyright: ignore ReportPrivateImportUsage
                    self._critic.parameters(), self._grad_clip
                )
            self._critic_optimizer.step()
            if self._log_schedule(global_step):
                logger.log_scalar("critic_loss", critic_loss, self.id)

            # Update policy with policy delay
            if self._policy_update_schedule(global_step):
                actor_loss = -torch.mean(
                    self._critic.q1(
                        current_state_inputs, self._actor(current_state_inputs)
                    )
                )
                self._actor_optimizer.zero_grad()
                actor_loss.backward()
                if self._grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(  # pyright: ignore ReportPrivateImportUsage
                        self._actor.parameters(), self._grad_clip
                    )
                self._actor_optimizer.step()
                self._update_target()
                if self._log_schedule(global_step):
                    logger.log_scalar("actor_loss", actor_loss, self.id)
        return agent_traj_state

    def _update_target(self):
        """Update the target network."""
        for network, target_network in [
            (self._actor, self._target_actor),
            (self._critic, self._target_critic),
        ]:
            target_params = target_network.state_dict()
            current_params = network.state_dict()
            for key in list(target_params.keys()):
                target_params[key] = (1 - self._soft_update_fraction) * target_params[
                    key
                ] + self._soft_update_fraction * current_params[key]
            target_network.load_state_dict(target_params)

    def save(self, dname):
        torch.save(
            {
                "critic": self._critic.state_dict(),
                "target_critic": self._target_critic.state_dict(),
                "critic_optimizer": self._critic_optimizer.state_dict(),
                "actor": self._actor.state_dict(),
                "target_actor": self._target_actor.state_dict(),
                "actor_optimizer": self._actor_optimizer.state_dict(),
            },
            os.path.join(dname, "agent.pt"),
        )
        replay_dir = os.path.join(dname, "replay")
        create_folder(replay_dir)
        self._replay_buffer.save(replay_dir)

    def load(self, dname):
        checkpoint = torch.load(os.path.join(dname, "agent.pt"))
        self._critic.load_state_dict(checkpoint["critic"])
        self._target_critic.load_state_dict(checkpoint["target_critic"])
        self._critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self._actor.load_state_dict(checkpoint["actor"])
        self._target_actor.load_state_dict(checkpoint["target_actor"])
        self._actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self._replay_buffer.load(os.path.join(dname, "replay"))
