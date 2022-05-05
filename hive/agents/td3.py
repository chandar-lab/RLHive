import copy
import os

import numpy as np
import torch
from torch.nn.functional import mse_loss

from hive.agents.agent import Agent
from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.td3_heads import TD3ActorNetwork, TD3CriticNetwork
from hive.agents.qnets.utils import (
    InitializationFn,
    calculate_output_dim,
    create_init_weights_fn,
)
from hive.replays import BaseReplayBuffer, CircularReplayBuffer
from hive.utils.loggers import Logger, NullLogger
from hive.utils.schedule import PeriodicSchedule, SwitchSchedule
from hive.utils.utils import LossFn, OptimizerFn, create_folder


class TD3(Agent):
    def __init__(
        self,
        observation_space,
        action_space,
        representation_net: FunctionApproximator = None,
        actor_net: FunctionApproximator = None,
        critic_net: FunctionApproximator = None,
        init_fn: InitializationFn = None,
        actor_optimizer_fn: OptimizerFn = None,
        critic_optimizer_fn: OptimizerFn = None,
        critic_loss_fn: LossFn = None,
        n_critics: int = 2,
        stack_size: int = 1,
        replay_buffer: BaseReplayBuffer = None,
        discount_rate: float = 0.99,
        n_step: int = 1,
        grad_clip: float = None,
        reward_clip: float = None,
        soft_update_fraction: float = 0.005,
        batch_size: int = 64,
        logger: Logger = None,
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
        super().__init__(observation_space, action_space, id)
        self._device = torch.device("cpu" if not torch.cuda.is_available() else device)
        self._state_size = (
            stack_size * self._observation_space.shape[0],
            *self._observation_space.shape[1:],
        )
        self._action_min = self._action_space.low
        self._action_max = self._action_space.high
        self._action_scaling = 0.5 * (self._action_max - self._action_min)
        self._scale_actions = np.isfinite(self._action_scaling).all()
        self._init_fn = create_init_weights_fn(init_fn)
        self._n_critics = n_critics
        self.create_networks(representation_net, actor_net, critic_net)
        if critic_optimizer_fn is None:
            critic_optimizer_fn = torch.optim.Adam
        if actor_optimizer_fn is None:
            actor_optimizer_fn = torch.optim.Adam
        self._critic_optimizer = critic_optimizer_fn(self._critic.parameters())
        self._actor_optimizer = actor_optimizer_fn(self._actor.parameters())
        if replay_buffer is None:
            replay_buffer = CircularReplayBuffer
        self._replay_buffer = replay_buffer(
            observation_shape=self._observation_space.shape,
            observation_dtype=self._observation_space.dtype,
            action_shape=self._action_space.shape,
            action_dtype=self._action_space.dtype,
            gamma=discount_rate,
        )
        self._discount_rate = discount_rate**n_step
        self._grad_clip = grad_clip
        self._reward_clip = reward_clip
        self._soft_update_fraction = soft_update_fraction
        if critic_loss_fn is None:
            critic_loss_fn = torch.nn.MSELoss
        self._critic_loss_fn = critic_loss_fn(reduction="none")
        self._batch_size = batch_size
        self._logger = logger
        if self._logger is None:
            self._logger = NullLogger([])
        self._timescale = self.id
        self._logger.register_timescale(
            self._timescale, PeriodicSchedule(False, True, log_frequency)
        )
        self._update_schedule = PeriodicSchedule(False, True, update_frequency)
        self._policy_update_schedule = PeriodicSchedule(
            False, True, policy_update_frequency
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
        """
        if representation_net is None:
            network = torch.nn.Identity()
        else:
            network = representation_net(self._state_size)
        network_output_dim = calculate_output_dim(network, self._state_size)
        self._actor = TD3ActorNetwork(
            network, actor_net, network_output_dim, self._action_space.shape
        ).to(self._device)
        self._critic = TD3CriticNetwork(
            network,
            critic_net,
            network_output_dim,
            self._n_critics,
            self._action_space.shape,
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
        if self._scale_actions:
            return ((actions - self._action_min) / self._action_scaling) - 1.0
        else:
            return actions

    def unscale_actions(self, actions):
        if self._scale_actions:
            return ((actions + 1.0) * self._action_scaling) + self._action_min
        else:
            return actions

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
            "action": self.scale_action(update_info["action"]),
            "reward": update_info["reward"],
            "done": update_info["done"],
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
    def act(self, observation):
        """Returns the action for the agent. If in training mode, adds noise with
        standard deviation :py:obj:`self._action_noise`.

        Args:
            observation: The current observation.
        """

        # Calculate action and add noise if training.
        observation = torch.tensor(
            np.expand_dims(observation, axis=0), device=self._device
        ).float()
        action = self._actor(observation)
        if self._training:
            noise = torch.randn_like(action, requires_grad=False) * self._action_noise
            action = action + noise
        action = action.cpu().detach().numpy()
        action = self.unscale_actions(np.expand_dims(action, axis=0))
        action = np.clip(action, self._action_min, self._action_max)
        return action

    def update(self, update_info):
        """
        Updates the TD3 agent.

        Args:
            update_info: dictionary containing all the necessary information to
                update the agent. Should contain a full transition, with keys for
                "observation", "action", "reward", and "done".
        """

        if not self._training:
            return

        # Add the most recent transition to the replay buffer.
        self._replay_buffer.add(**self.preprocess_update_info(update_info))

        # Update the agent based on a sample batch from the replay buffer.
        if (
            self._learn_schedule.update()
            and self._replay_buffer.size() > 0
            and self._update_schedule.update()
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
                next_actions = torch.clamp(
                    self._target_actor(*next_state_inputs) + noise, -1, 1
                )
                next_q_vals = torch.cat(
                    self._target_critic(*next_state_inputs, next_actions), dim=1
                )
                next_q_vals, _ = torch.min(next_q_vals, dim=1, keepdim=True)
                target_q_values = (
                    batch["reward"][:, None]
                    + (1 - batch["done"][:, None]) * self._discount_rate * next_q_vals
                )

            # Critic losses
            pred_qvals = self._critic(*current_state_inputs, batch["action"])
            critic_loss = sum(
                [mse_loss(qvals, target_q_values) for qvals in pred_qvals]
            )
            self._critic_optimizer.zero_grad()
            critic_loss.backward()
            if self._grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self._critic.parameters(), self._grad_clip
                )
            self._critic_optimizer.step()
            if self._logger.update_step(self._timescale):
                self._logger.log_scalar("critic_loss", critic_loss, self._timescale)

            # Update policy with policy delay
            if self._policy_update_schedule.update():
                actor_loss = -torch.mean(
                    self._critic.q1(
                        *current_state_inputs, self._actor(*current_state_inputs)
                    )
                )
                self._actor_optimizer.zero_grad()
                actor_loss.backward()
                if self._grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self._actor.parameters(), self._grad_clip
                    )
                self._actor_optimizer.step()
                self._update_target()
                if self._logger.should_log(self._timescale):
                    self._logger.log_scalar("actor_loss", actor_loss, self._timescale)

    def _update_target(self):
        """Update the target network."""
        for (network, target_network) in [
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
                "learn_schedule": self._learn_schedule,
                "update_schedule": self._update_schedule,
                "policy_update_schedule": self._policy_update_schedule,
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
        self._learn_schedule = checkpoint["learn_schedule"]
        self._update_schedule = checkpoint["update_schedule"]
        self._policy_update_schedule = checkpoint["policy_update_schedule"]
        self._replay_buffer.load(os.path.join(dname, "replay"))
