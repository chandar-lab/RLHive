import os

import gym
import numpy as np
import torch
from typing import Union

from hive.agents.agent import Agent
from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.ppo_nets import PPOActorNetwork, PPOCriticNetwork
from hive.agents.qnets.utils import (
    InitializationFn,
    calculate_output_dim,
    create_init_weights_fn,
)
from hive.replays.ppo_replay import PPOReplayBuffer
from hive.utils.loggers import Logger, NullLogger
from hive.utils.schedule import (
    PeriodicSchedule,
)
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
        init_fn: InitializationFn = None,
        actor_optimizer_fn: OptimizerFn = None,
        critic_optimizer_fn: OptimizerFn = None,
        critic_loss_fn: LossFn = None,
        stack_size: int = 1,
        replay_buffer: PPOReplayBuffer = None,
        discount_rate: float = 0.99,
        n_step: int = 1,
        grad_clip: float = None,
        reward_clip: float = None,
        batch_size: int = 64,
        logger: Logger = None,
        log_frequency: int = 100,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        clip_vloss: bool = True,
        vf_coef: float = 0.5,
        num_epochs_per_update: int = 4,
        normalize_advantages: bool = True,
        target_kl=None,
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
        self._state_size = (
            stack_size * self._observation_space.shape[0],
            *self._observation_space.shape[1:],
        )
        self._continuous_action = isinstance(self._action_space, gym.spaces.Box)
        self._init_fn = create_init_weights_fn(init_fn)
        self.create_networks(representation_net, actor_net, critic_net)
        if actor_optimizer_fn is None:
            actor_optimizer_fn = torch.optim.Adam
        self._actor_optimizer = actor_optimizer_fn(self._actor.parameters())
        if critic_optimizer_fn is None:
            critic_optimizer_fn = torch.optim.Adam
        self._critic_optimizer = critic_optimizer_fn(self._critic.parameters())
        if replay_buffer is None:
            replay_buffer = PPOReplayBuffer
        extra_storage_types = {
            "values": (np.float32, ()),
            "returns": (np.float32, ()),
            "advantages": (np.float32, ()),
            "logprob": (np.float32, ()),
        }
        self._replay_buffer = replay_buffer(
            observation_shape=self._observation_space.shape,
            observation_dtype=self._observation_space.dtype,
            action_shape=self._action_space.shape,
            action_dtype=self._action_space.dtype,
            gamma=discount_rate,
            extra_storage_types=extra_storage_types,
        )
        self._discount_rate = discount_rate**n_step
        self._grad_clip = grad_clip
        self._reward_clip = reward_clip
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
        self._clip_coef = clip_coef
        self._ent_coef = ent_coef
        self._clip_vloss = clip_vloss
        self._vf_coef = vf_coef
        self._num_epochs = num_epochs_per_update
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
        self._actor = PPOActorNetwork(
            network,
            actor_net,
            network_output_shape,
            self._action_space,
            self._continuous_action,
        ).to(self._device)
        self._critic = PPOCriticNetwork(
            network,
            critic_net,
            network_output_shape,
        ).to(self._device)

        self._actor.apply(self._init_fn)
        self._critic.apply(self._init_fn)

    def train(self):
        """Changes the agent to training mode."""
        super().train()
        self._actor.train()
        self._critic.train()

    def eval(self):
        """Changes the agent to evaluation mode."""
        super().eval()
        self._actor.eval()
        self._critic.eval()

    def preprocess_update_info(self, update_info):
        """Preprocesses the :obj:`update_info` before it goes into the replay buffer.

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
            "action": update_info["action"],
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
                - Preprocessed batch.
        """
        for key in batch:
            batch[key] = torch.tensor(batch[key], device=self._device)
        return batch

    @torch.no_grad()
    def act(self, observation):
        """Returns the action for the agent.

        Args:
            observation: The current observation.
        """

        observation = torch.tensor(
            np.expand_dims(observation, axis=0), device=self._device
        ).float()
        action, logprob, _ = self._actor(observation)
        value = self._critic(observation)
        action = action.cpu().detach().numpy()
        action = action[0]

        self._logprob = logprob.cpu().numpy()
        self._value = value.cpu().numpy()
        return action

    def update(self, update_info):
        """
        Updates the PPO agent.

        Args:
            update_info: dictionary containing all the necessary information to
                update the agent. Should contain a full transition, with keys for
                "observation", "action", "reward", and "done".
        """
        if not self._training:
            return
        if self._replay_buffer.ready():
            self._replay_buffer.compute_advantages(self._value)
            for _ in range(self._num_epochs):
                valid_ind_size = self._replay_buffer._find_valid_indices()
                for _ in range(valid_ind_size // self._batch_size):
                    batch = self._replay_buffer.sample(batch_size=self._batch_size)
                    batch = self.preprocess_update_batch(batch)
                    self._actor_optimizer.zero_grad()
                    self._critic_optimizer.zero_grad()

                    _, _logprob, entropy = self._actor(
                        batch["observation"], batch["action"]
                    )
                    values = self._critic(batch["observation"])
                    logratios = _logprob - batch["logprob"]
                    ratios = torch.exp(logratios)
                    advantages = batch["advantages"]
                    if self._normalize_advantages:
                        advantages = (advantages - advantages.mean()) / (
                            advantages.std() + 1e-8
                        )
                    # Actor loss
                    loss1 = -advantages * ratios
                    loss2 = -advantages * torch.clamp(
                        ratios, 1 - self._clip_coef, 1 + self._clip_coef
                    )
                    actor_loss = torch.max(loss1, loss2).mean()
                    entr_loss = entropy.mean()

                    # Critic loss
                    values = values.view(-1)
                    if self._clip_vloss:
                        v_loss_unclipped = self._critic_loss_fn(
                            values, batch["returns"]
                        )
                        v_clipped = batch["values"] + torch.clamp(
                            values - batch["values"],
                            -self._clip_coef,
                            self._clip_coef,
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
                        - self._ent_coef * entr_loss
                        + self._vf_coef * critic_loss
                    )
                    loss.backward()

                    if self._grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self._critic.parameters(), self._grad_clip
                        )
                        torch.nn.utils.clip_grad_norm_(
                            self._actor.parameters(), self._grad_clip
                        )

                    self._actor_optimizer.step()
                    self._critic_optimizer.step()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        approx_kl = ((ratios - 1) - logratios).mean()

                    if self._logger.should_log(self._timescale):
                        self._logger.log_scalar(
                            "actor_loss", actor_loss, self._timescale
                        )
                        self._logger.log_scalar(
                            "critic_loss", critic_loss, self._timescale
                        )
                        self._logger.log_scalar(
                            "entropy_loss", entr_loss, self._timescale
                        )
                        self._logger.log_scalar("approxkl", approx_kl, self._timescale)

                if self._target_kl is not None:
                    if approx_kl > self._target_kl:
                        break
            self._replay_buffer.reset()

        processed_update_info = self.preprocess_update_info(update_info)
        processed_update_info.update(
            {
                "logprob": self._logprob,
                "values": self._value,
                "returns": np.empty(self._value.shape),
                "advantages": np.empty(self._value.shape),
            }
        )
        self._replay_buffer.add(**processed_update_info)

    def save(self, dname):
        torch.save(
            {
                "critic": self._critic.state_dict(),
                "actor_optimizer": self._actor_optimizer.state_dict(),
                "actor": self._actor.state_dict(),
                "critic_optimizer": self._critic_optimizer.state_dict(),
            },
            os.path.join(dname, "agent.pt"),
        )
        replay_dir = os.path.join(dname, "replay")
        create_folder(replay_dir)
        self._replay_buffer.save(replay_dir)

    def load(self, dname):
        checkpoint = torch.load(os.path.join(dname, "agent.pt"))
        self._actor.load_state_dict(checkpoint["actor"])
        self._critic.load_state_dict(checkpoint["critic"])
        self._actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self._critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self._replay_buffer.load(os.path.join(dname, "replay"))
