import os
from turtle import update

import gym
import numpy as np
import torch

from hive.agents.agent import Agent
from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.qnet_heads import DQNNetwork
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
from hive.utils.utils import LossFn, OptimizerFn, create_folder, seeder
from hive.agents.qnets.ppo_nets import PPOActorCriticNetwork

class PPOAgent(Agent):
    """An agent implementing the PPO algorithm. Uses an epsilon greedy
    exploration policy
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        representation_net: FunctionApproximator,
        actor_net: FunctionApproximator = None,
        critic_net: FunctionApproximator = None,
        stack_size: int = 1,
        id=0,
        init_fn: InitializationFn = None,
        optimizer_fn: OptimizerFn = None,
        replay_buffer: PPOReplayBuffer = None,
        discount_rate: float = 0.99,
        n_step: int = 1,
        transitions_per_update = 100, 
        grad_norm_clip: float = 0.5,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        clip_vloss: bool = True,
        normalize_advantages: bool = True,
        gae_lambda: float = 0.95,
        num_updates: int = 4,
        batch_size: int = 32,
        device="cpu",
        logger: Logger = None,
        log_frequency: int = 100,
    ):
        super().__init__(
            observation_space=observation_space, action_space=action_space, id=id
        )
        self._state_size = (
            stack_size * self._observation_space.shape[0],
            *self._observation_space.shape[1:],
        )

        self._device = torch.device("cpu" if not torch.cuda.is_available() else device)
        
        # self._action_min = torch.tensor(self._action_space.low)
        # self._action_max = torch.tensor(self._action_space.high)
        # self._action_scaling = 0.5 * (self._action_max - self._action_min)
        
        self.create_networks(representation_net, actor_net, critic_net)
        
        self._init_fn = create_init_weights_fn(init_fn)
        
        if optimizer_fn is None:
            optimizer_fn = torch.optim.Adam
        self._optimizer = optimizer_fn(self._actor_critic.parameters())

        if replay_buffer is None:
            replay_buffer = PPOReplayBuffer

        extra_storage_types = {
            "value": (np.float32, ()), 
            "return": (np.float32, ()),
            "advantage": (np.float32, ()),
            "logprob": (np.float32, self._action_space.shape),
        }
        self._replay_buffer = replay_buffer(
            observation_shape=self._observation_space.shape,
            observation_dtype=self._observation_space.dtype,
            action_shape=self._action_space.shape,
            action_dtype=self._action_space.dtype,
            gamma=discount_rate,
            gae_lambda=gae_lambda,
            extra_storage_types=extra_storage_types
        )
        
        self._discount_rate = discount_rate**n_step
        
        self._transitions_per_update = transitions_per_update
        
        # if loss_fn is None:
        #     loss_fn = torch.nn.SmoothL1Loss
        # self._loss_fn = loss_fn(reduction="none")
        
        self._batch_size = batch_size

        self._logger = logger
        if self._logger is None:
            self._logger = NullLogger([])
        
        self._timescale = self.id
        self._logger.register_timescale(
            self._timescale, PeriodicSchedule(False, True, log_frequency)
        )

        self._state = {"episode_start": True}
        self._training = False
        self._num_epochs = num_updates
        
        self._clip_coef = clip_coef

        self._ent_coef = ent_coef
        self._vf_coef = vf_coef

        self._norm_adv = normalize_advantages
        self._clip_vloss = clip_vloss
        self._grad_norm_clip = grad_norm_clip
    
    def create_networks(self, representation_net, actor_net, critic_net):
        """Creates the actor and critic networks.

        Args:
            representation_net: A network that outputs the shared representations that
                will be used by the actor and critic networks to process observations.
        """
        if representation_net is None:
            network = torch.nn.Identity()
        else:
            network = representation_net(self._observation_space.shape)
        network_output_dim = calculate_output_dim(
            network, self._observation_space.shape
        )
        self._actor_critic = PPOActorCriticNetwork(
            network, actor_net, critic_net, network_output_dim, self._action_space
        ).to(self._device)

        # self._actor_critic.apply(self._init_fn)

    def train(self):
        """Changes the agent to training mode."""
        super().train()
        self._actor_critic.train()

    def eval(self):
        """Changes the agent to evaluation mode."""
        super().eval()
        self._actor_critic.eval()

    def preprocess_update_info(self, update_info):
        """Preprocesses the :obj:`update_info` before it goes into the replay buffer.
        Clips the reward in update_info.

        Args:
            update_info: Contains the information from the current timestep that the
                agent should use to update itself.
        """
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
        action, logprob, _, value = self._actor_critic.get_action_value(observation)
        action = action.item()
        if (
            self._training
            and self._logger.should_log(self._timescale)
            and self._state["episode_start"]
        ):
            self._logger.log_scalar("train_qval", value, self._timescale)
            self._state["episode_start"] = False
        
        
        self._logprob = logprob
        self._cur_value = value
        return action

    def update(self, update_info):
        """
        Updates the PPO agent.

        Args:
            update_info: dictionary containing all the necessary information to
                update the agent. Should contain a full transition, with keys for
                "observation", "action", "reward", and "done".
        """
        
        if update_info["done"]:
            self._state["episode_start"] = True

        if not self._training:
            return
        
        # Inspired  by https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_shared.py
        # Check if the buffer is already full, if it is use current observation to bootstrap value estimate
        if (self._replay_buffer.size()+1 >= self._replay_buffer._capacity): #We need next observation

            observation = torch.tensor(
                    np.expand_dims(update_info["observation"], axis=0), device=self._device
                ).float()
            with torch.no_grad():
                values = self._actor_critic.get_value(observation)
            self._replay_buffer.compute_advantages(values.cpu().numpy(), update_info["done"])

            for _ in range(self._num_epochs):
                valid_ind_size = self._replay_buffer._find_valid_random()

                #Start sampling again from scratch TODO
                for _ in range(valid_ind_size//self._batch_size):
                    # Check if it samples without replacement and samples copmletely? TODO
                    batch = self._replay_buffer.sample(batch_size=self._batch_size)
                    batch = self.preprocess_update_batch(batch)
                    self._optimizer.zero_grad()

                    _, newlogprob, entropy, newvalue = self._actor_critic.get_action_value(batch["observation"], batch["action"])
                    logratio = newlogprob - batch["logprob"]
                    ratio = logratio.exp()
                    
                    advantages = batch["advantage"]
                    if self._norm_adv:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -advantages * ratio
                    pg_loss2 = -advantages * torch.clamp(ratio, 1 - self._clip_coef, 1 + self._clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self._clip_vloss:
                        v_loss_unclipped = (newvalue - batch["return"]) ** 2
                        v_clipped = batch["value"] + torch.clamp(
                            newvalue - batch["value"],
                            -self._clip_coef,
                            self._clip_coef,
                        )
                        v_loss_clipped = (v_clipped - batch["return"]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - batch["return"]) ** 2).mean()

                    # Entropy Loss
                    entropy_loss = entropy.mean()
                    approxkl = (-logratio).mean()
                    loss = pg_loss - self._ent_coef * entropy_loss + v_loss * self._vf_coef


                    if self._logger.should_log(self._timescale):
                        self._logger.log_scalar("train_loss", loss, self._timescale)

                        #added for debugging
                        self._logger.log_scalar("policy_loss", pg_loss, self._timescale)
                        self._logger.log_scalar("value_loss", v_loss, self._timescale)
                        self._logger.log_scalar("entropy_loss", entropy_loss, self._timescale)
                        self._logger.log_scalar("clip_frac", (torch.abs(ratio-1.0) > self._clip_coef).float().mean(), self._timescale)
                        self._logger.log_scalar("approxkl", approxkl, self._timescale)
                    
                    loss.backward()
                    if self._grad_norm_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self._actor_critic.parameters(), self._grad_norm_clip
                        )
                    self._optimizer.step()
            
            self._replay_buffer.reset()
        
        processed_update_info = self.preprocess_update_info(update_info)
        processed_update_info.update({
            'logprob':self._logprob.cpu().numpy(),
            'value':self._cur_value.cpu().numpy(),
            'return':np.empty(self._cur_value.shape),
            'advantage':np.empty(self._cur_value.shape),
        })
        # Add the most recent transition to the replay buffer.
        self._replay_buffer.add(**processed_update_info)
        

    def save(self, dname):
        torch.save(
            {
                "ac_model": self._actor_critic.state_dict(),
                "optimizer": self._optimizer.state_dict(),
            },
            os.path.join(dname, "agent.pt"),
        )
        replay_dir = os.path.join(dname, "replay")
        create_folder(replay_dir)
        self._replay_buffer.save(replay_dir)

    def load(self, dname):
        checkpoint = torch.load(os.path.join(dname, "agent.pt"))
        self._ac_model.load_state_dict(checkpoint["ac_model"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])
        self._replay_buffer.load(os.path.join(dname, "replay"))
