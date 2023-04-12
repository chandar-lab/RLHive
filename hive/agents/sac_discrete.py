import copy

import gymnasium as gym
import numpy as np
import torch

from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.utils import (
    InitializationFn,
    calculate_output_dim,
)
from hive.replays import BaseReplayBuffer
from hive.utils.loggers import Logger
from hive.utils.utils import LossFn, OptimizerFn
from hive.agents.qnets.sac_heads import SACActorNetwork, SACDiscreteCriticNetwork


class DiscreteSACAgent(SACAgent):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
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
        min_replay_history: int = 1000,
        auto_alpha: bool = True,
        alpha: float = 0.2,
        target_entropy_scale: float = 0.98,
        alpha_optimizer_fn: OptimizerFn = None,
        device="cpu",
        id=0,
    ):
        self._target_entropy_scale = target_entropy_scale
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            representation_net=representation_net,
            actor_net=actor_net,
            critic_net=critic_net,
            init_fn=init_fn,
            actor_optimizer_fn=actor_optimizer_fn,
            critic_optimizer_fn=critic_optimizer_fn,
            critic_loss_fn=critic_loss_fn,
            n_critics=n_critics,
            stack_size=stack_size,
            replay_buffer=replay_buffer,
            discount_rate=discount_rate,
            n_step=n_step,
            grad_clip=grad_clip,
            reward_clip=reward_clip,
            soft_update_fraction=soft_update_fraction,
            batch_size=batch_size,
            logger=logger,
            log_frequency=log_frequency,
            update_frequency=update_frequency,
            policy_update_frequency=policy_update_frequency,
            min_replay_history=min_replay_history,
            auto_alpha=auto_alpha,
            alpha=alpha,
            alpha_optimizer_fn=alpha_optimizer_fn,
            device=device,
            id=id,
        )

    def create_networks(self, representation_net, actor_net, critic_net):
        """Creates the actor and critic networks.

        Args:
            representation_net: A network that outputs the shared representations that
                will be used by the actor and critic networks to process observations.
            actor_net: The network that will be used to compute actions.
            critic_net: The network that will be used to compute values of state action
                pairs.
        """
        if representation_net is None:
            network = torch.nn.Identity()
        else:
            network = representation_net(self._state_size)
        network_output_shape = calculate_output_dim(network, self._state_size)
        self._actor = SACActorNetwork(
            network,
            actor_net,
            network_output_shape,
            self._action_space,
        ).to(self._device)
        self._critic = SACDiscreteCriticNetwork(
            network,
            critic_net,
            network_output_shape,
            self._action_space,
            self._n_critics,
        ).to(self._device)

        self._actor.apply(self._init_fn)
        self._critic.apply(self._init_fn)
        self._target_critic = copy.deepcopy(self._critic).requires_grad_(False)
        # Automatic entropy tuning
        if self._auto_alpha:
            self._target_entropy = self._target_entropy_scale * np.log(
                self._action_space.n
            )
            self._log_alpha = torch.zeros(1, requires_grad=True, device=self._device)

    def _update_actor(self, current_state_inputs):
        _, log_probs, action_probs = self._actor(*current_state_inputs)
        with torch.no_grad():
            action_values = self._critic(*current_state_inputs)
        min_action_values = torch.amin(torch.stack(action_values), dim=0)
        actor_loss = torch.mean(
            action_probs * (self._alpha * log_probs - min_action_values)
        )
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        if self._grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self._actor.parameters(), self._grad_clip)
        self._actor_optimizer.step()
        if self._auto_alpha:
            alpha_loss = self._update_alpha(log_probs, action_probs)
        else:
            alpha_loss = 0
        return actor_loss, alpha_loss

    def _update_alpha(self, log_probs, action_probs):
        alpha_loss = (
            action_probs.detach()
            * (-self._log_alpha * (log_probs + self._target_entropy).detach())
        ).mean()

        self._alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self._alpha_optimizer.step()
        self._alpha = self._log_alpha.exp().item()
        return alpha_loss

    def _update_critics(self, batch, current_state_inputs, next_state_inputs):
        target_q_values = self._calculate_target_q_values(batch, next_state_inputs)

        # Critic losses
        pred_qvals = torch.stack(self._critic(*current_state_inputs))
        pred_qvals = pred_qvals[:, torch.arange(pred_qvals.shape[1]), batch["action"]]
        critic_loss = sum(
            [self._critic_loss_fn(qvals, target_q_values) for qvals in pred_qvals]
        )
        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        if self._grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self._critic.parameters(), self._grad_clip)
        self._critic_optimizer.step()
        return critic_loss

    def _calculate_target_q_values(self, batch, next_state_inputs):
        with torch.no_grad():
            _, next_log_prob, next_probs = self._actor(*next_state_inputs)
            next_q_vals = torch.stack(self._target_critic(*next_state_inputs))
            next_q_vals = torch.amin(next_q_vals, dim=0)
            next_q_vals = next_probs * (next_q_vals - self._alpha * next_log_prob)
            next_q_vals = torch.sum(next_q_vals, dim=1)
            target_q_values = (
                batch["reward"]
                + (1 - batch["terminated"]) * self._discount_rate * next_q_vals
            )

        return target_q_values
