import os
from typing import Union

import gymnasium as gym
import numpy as np
import torch

from hive.agents.agent import Agent
from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.normalizer import (
    MovingAvgNormalizer,
    RewardNormalizer,
)
from hive.agents.qnets.ac_nets import ActorCriticNetwork
from hive.agents.qnets.utils import calculate_output_dim
from hive.replays.on_policy_replay import OnPolicyReplayBuffer
from hive.utils.loggers import Logger, NullLogger
from hive.utils.schedule import PeriodicSchedule, Schedule, ConstantSchedule
from hive.utils.utils import LossFn, OptimizerFn, create_folder


class PPOAgent(Agent):
    """An agent implementing the PPO algorithm."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: Union[gym.spaces.Discrete, gym.spaces.Box],
        representation_net: FunctionApproximator = None,
        actor_net: FunctionApproximator = None,
        critic_net: FunctionApproximator = None,
        optimizer_fn: OptimizerFn = None,
        anneal_lr_schedule: Schedule = None,
        critic_loss_fn: LossFn = None,
        observation_normalizer: MovingAvgNormalizer = None,
        reward_normalizer: RewardNormalizer = None,
        stack_size: int = 1,
        replay_buffer: OnPolicyReplayBuffer = None,
        discount_rate: float = 0.99,
        n_step: int = 1,
        grad_clip: float = None,
        batch_size: int = 64,
        logger: Logger = None,
        log_frequency: int = 1,
        clip_coefficient: float = 0.2,
        entropy_coefficient: float = 0.01,
        clip_value_loss: bool = True,
        value_fn_coefficient: float = 0.5,
        transitions_per_update: int = 1024,
        num_epochs_per_update: int = 4,
        normalize_advantages: bool = True,
        target_kl: float = None,
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
            optimizer_fn (OptimizerFn): A function that takes in the list of
                parameters of the actor and critic returns the optimizer for the actor.
                If None, defaults to :py:class:`~torch.optim.Adam`.
            critic_loss_fn (LossFn): The loss function used to optimize the critic. If
                None, defaults to :py:class:`~torch.nn.MSELoss`.
            observation_normalizer (MovingAvgNormalizer): The function for
                normalizing observations
            reward_normalizer (RewardNormalizer): The function for normalizing
                rewards
            stack_size (int): Number of observations stacked to create the state fed
                to the agent.
            replay_buffer (OnPolicyReplayBuffer): The replay buffer that the agent will
                push observations to and sample from during learning. If None,
                defaults to
                :py:class:`~hive.replays.circular_replay.OnPolicyReplayBuffer`.
            discount_rate (float): A number between 0 and 1 specifying how much
                future rewards are discounted by the agent.
            n_step (int): The horizon used in n-step returns to compute TD(n) targets.
            grad_clip (float): Gradients will be clipped to between
                [-grad_clip, grad_clip].
            batch_size (int): The size of the batch sampled from the replay buffer
                during learning.
            logger (Logger): Logger used to log agent's metrics.
            log_frequency (int): How often to log the agent's metrics.
            clip_coefficient (float): A number between 0 and 1 specifying the clip ratio
                for the surrogate objective function to penalise large changes in
                the policy and/or critic.
            entropy_coefficient (float): Coefficient for the entropy loss.
            clip_value_loss (bool): Flag to use the clipped objective for the value
                function.
            value_fn_coefficient (float): Coefficient for the value function loss.
            transitions_per_update (int): Total number of observations that are
                stored before the update.
            num_epochs_per_update (int): Number of iterations over the entire
                buffer during an update step.
            normalize_advantages (bool): Flag to normalise advantages before
                calculating policy loss.
            target_kl (float): Terminates the update if kl-divergence between old and
                updated policy exceeds target_kl.
            device: Device on which all computations should be run.
            id: Agent identifier.
        """
        super().__init__(observation_space, action_space, id)
        self._device = torch.device("cpu" if not torch.cuda.is_available() else device)
        self._state_size = (
            stack_size * self._observation_space.shape[0],
            *self._observation_space.shape[1:],
        )
        self.create_networks(
            representation_net,
            actor_net,
            critic_net,
        )
        if observation_normalizer is not None:
            self._observation_normalizer = observation_normalizer(self._state_size)
        else:
            self._observation_normalizer = None

        if reward_normalizer is not None:
            self._reward_normalizer = reward_normalizer(discount_rate)
        else:
            self._reward_normalizer = None

        if optimizer_fn is None:
            optimizer_fn = torch.optim.Adam
        self._optimizer = optimizer_fn(self._actor_critic.parameters())
        if anneal_lr_schedule is None:
            anneal_lr_schedule = ConstantSchedule(1.0)
        else:
            anneal_lr_schedule = anneal_lr_schedule()

        self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._optimizer, lambda x: anneal_lr_schedule.update()
        )
        if replay_buffer is None:
            replay_buffer = OnPolicyReplayBuffer
        self._replay_buffer = replay_buffer(
            capacity=transitions_per_update,
            observation_shape=self._observation_space.shape,
            observation_dtype=self._observation_space.dtype,
            action_shape=self._action_space.shape,
            action_dtype=self._action_space.dtype,
            gamma=discount_rate,
        )
        self._discount_rate = discount_rate**n_step
        self._grad_clip = grad_clip
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
        self._clip_coefficient = clip_coefficient
        self._entropy_coefficient = entropy_coefficient
        self._clip_value_loss = clip_value_loss
        self._value_fn_coefficient = value_fn_coefficient
        self._transitions_per_update = transitions_per_update
        self._num_epochs_per_update = num_epochs_per_update
        self._normalize_advantages = normalize_advantages
        self._target_kl = target_kl

        self._training = False

    def create_networks(self, representation_net, actor_net, critic_net):
        """Creates the actor and critic networks.

        Args:
            representation_net: A network that outputs the shared representations that
                will be used by the actor and critic networks to process observations.
            actor_net: The network that will be used to compute actions.
            critic_net: The network that will be used to compute values of states.
        """
        if representation_net is None:
            network = torch.nn.Identity()
        else:
            network = representation_net(self._state_size)

        network_output_shape = calculate_output_dim(network, self._state_size)
        self._actor_critic = ActorCriticNetwork(
            network,
            actor_net,
            critic_net,
            network_output_shape,
            self._action_space,
            isinstance(self._action_space, gym.spaces.Box),
        ).to(self._device)

    def train(self):
        """Changes the agent to training mode."""
        super().train()
        self._actor_critic.train()

    def eval(self):
        """Changes the agent to evaluation mode."""
        super().eval()
        self._actor_critic.eval()

    def preprocess_update_info(self, update_info, agent_traj_state):
        """Preprocesses the :obj:`update_info` before it goes into the replay buffer.

        Args:
            update_info: Contains the information from the current timestep that the
                agent should use to update itself.
        """
        if self._observation_normalizer:
            update_info["observation"] = self._observation_normalizer(
                update_info["observation"]
            )

        done = update_info["terminated"] or update_info["truncated"]
        if self._reward_normalizer:
            self._reward_normalizer.update(update_info["reward"], done)
            update_info["reward"] = self._reward_normalizer(update_info["reward"])

        preprocessed_update_info = {
            "observation": update_info["observation"],
            "action": update_info["action"],
            "reward": update_info["reward"],
            "done": done,
            "logprob": agent_traj_state["logprob"],
            "values": agent_traj_state["value"],
            "returns": np.empty(agent_traj_state["value"].shape),
            "advantages": np.empty(agent_traj_state["value"].shape),
        }
        if "agent_id" in update_info:
            preprocessed_update_info["agent_id"] = int(update_info["agent_id"])

        return preprocessed_update_info

    def preprocess_update_batch(self, batch):
        """Returns preprocesed batch sampled from the replay buffer.

        Args:
            batch: Batch sampled from the replay buffer for the current update.
        """
        for key in batch:
            batch[key] = torch.tensor(batch[key], device=self._device)

        return batch

    @torch.no_grad()
    def get_action_logprob_value(self, observation):
        """Returns the action, logprob, and value for the agent

        Args:
            observation: The current observation.
        """
        observation = torch.tensor(
            np.expand_dims(observation, axis=0), device=self._device
        ).float()
        action, logprob, _, value = self._actor_critic(observation)
        action = action.cpu().detach().numpy()
        logprob = logprob.cpu().numpy()
        value = value.cpu().numpy()
        action = action[0]

        return action, logprob, value

    @torch.no_grad()
    def act(self, observation, agent_traj_state=None):
        """Returns the action for the agent.

        Args:
            observation: The current observation.
        """
        if agent_traj_state is None:
            agent_traj_state = {}
        if self._observation_normalizer:
            self._observation_normalizer.update(observation)
            observation = self._observation_normalizer(observation)
        action, logprob, value = self.get_action_logprob_value(observation)
        agent_traj_state["logprob"] = logprob
        agent_traj_state["value"] = value
        return action, agent_traj_state

    def update(self, update_info, agent_traj_state=None):
        """
        Updates the PPO agent.

        Args:
            update_info: dictionary containing all the necessary information to
                update the agent. Should contain a full transition, with keys for
                "observation", "next_observation", "action", "reward", "terminated",
                and "truncated".
        """
        if not self._training:
            return

        # Add the most recent transition to the replay buffer.
        self._replay_buffer.add(
            **self.preprocess_update_info(update_info, agent_traj_state)
        )

        if self._replay_buffer.size() >= self._transitions_per_update - 1:
            if self._observation_normalizer:
                update_info["next_observation"] = self._observation_normalizer(
                    update_info["next_observation"]
                )
            _, _, values = self.get_action_logprob_value(
                update_info["next_observation"]
            )
            self._replay_buffer.compute_advantages(values)
            clip_fraction = 0
            num_updates = 0
            for _ in range(self._num_epochs_per_update):
                for batch in self._replay_buffer.sample(batch_size=self._batch_size):
                    batch = self.preprocess_update_batch(batch)
                    self._optimizer.zero_grad()

                    _, logprob, entropy, values = self._actor_critic(
                        batch["observation"], batch["action"]
                    )
                    logratios = logprob - batch["logprob"]
                    ratios = torch.exp(logratios)
                    advantages = batch["advantages"]
                    if self._normalize_advantages:
                        advantages = (advantages - advantages.mean()) / (
                            advantages.std() + 1e-8
                        )
                    # Actor loss
                    loss_unclipped = -advantages * ratios
                    loss_clipped = -advantages * torch.clamp(
                        ratios, 1 - self._clip_coefficient, 1 + self._clip_coefficient
                    )
                    actor_loss = torch.max(loss_unclipped, loss_clipped).mean()
                    entropy_loss = entropy.mean()

                    # Critic loss
                    values = values.view(-1)
                    if self._clip_value_loss:
                        v_loss_unclipped = self._critic_loss_fn(
                            values, batch["returns"]
                        )
                        v_clipped = batch["values"] + torch.clamp(
                            values - batch["values"],
                            -self._clip_coefficient,
                            self._clip_coefficient,
                        )
                        v_loss_clipped = self._critic_loss_fn(
                            v_clipped, batch["returns"]
                        )
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        critic_loss = 0.5 * v_loss_max.mean()
                    else:
                        critic_loss = (
                            0.5 * self._critic_loss_fn(values, batch["returns"]).mean()
                        )

                    loss = (
                        actor_loss
                        - self._entropy_coefficient * entropy_loss
                        + self._value_fn_coefficient * critic_loss
                    )
                    loss.backward()

                    if self._grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self._actor_critic.parameters(), self._grad_clip
                        )

                    self._optimizer.step()
                    num_updates += 1

                    with torch.no_grad():
                        # calculate approx_kl
                        # http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratios).mean()
                        approx_kl = ((ratios - 1) - logratios).mean()
                        clip_fraction += (
                            ((ratios - 1.0).abs() > self._clip_coefficient)
                            .float()
                            .mean()
                            .item()
                        )

                if self._target_kl is not None and self._target_kl < approx_kl:
                    break
            self._replay_buffer.reset()
            if self._logger.update_step(self._timescale):
                self._logger.log_metrics(
                    {
                        "loss": loss,
                        "actor_loss": actor_loss,
                        "critic_loss": critic_loss,
                        "entropy_loss": entropy_loss,
                        "approx_kl": approx_kl,
                        "old_approx_kl": old_approx_kl,
                        "clip_fraction": clip_fraction / num_updates,
                        "lr": self._lr_scheduler.get_last_lr()[0],
                    },
                    prefix=self._timescale,
                )
            self._lr_scheduler.step()
        return agent_traj_state

    def save(self, dname):
        state_dict = {
            "actor_critic": self._actor_critic.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }
        if self._observation_normalizer:
            state_dict[
                "observation_normalizer"
            ] = self._observation_normalizer.state_dict()
        if self._reward_normalizer:
            state_dict["reward_normalizer"] = self._reward_normalizer.state_dict()
        torch.save(
            state_dict,
            os.path.join(dname, "agent.pt"),
        )
        replay_dir = os.path.join(dname, "replay")
        create_folder(replay_dir)
        self._replay_buffer.save(replay_dir)

    def load(self, dname):
        checkpoint = torch.load(os.path.join(dname, "agent.pt"))
        self._actor_critic.load_state_dict(checkpoint["actor_critic"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])
        self._replay_buffer.load(os.path.join(dname, "replay"))
        if self._observation_normalizer:
            self._observation_normalizer.load_state_dict(
                checkpoint["observation_normalizer"]
            )
        if self._reward_normalizer:
            self._reward_normalizer.load_state_dict(checkpoint["reward_normalizer"])
