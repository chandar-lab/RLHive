from typing import Optional

import gymnasium as gym
import torch

from hive.agents.td3.td3 import TD3
from hive.replays import BaseReplayBuffer
from hive.types import Creates, Partial
from hive.utils.torch_utils import ModuleInitFn
from hive.utils.utils import LossFn


class DDPG(TD3):
    """
    An agent implementing the DDPG algorithm. It is implemented by fixing the
    n_critics, policy_update_frequency, target_noise, and target_noise_clip parameters
    of the :py:class:`~hive.agents.td3.TD3` agent.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        representation_net: Optional[Creates[torch.nn.Module]] = None,
        actor_net: Optional[Creates[torch.nn.Module]] = None,
        critic_net: Optional[Creates[torch.nn.Module]] = None,
        init_fn: Optional[Partial[ModuleInitFn]] = None,
        actor_optimizer_fn: Optional[Creates[torch.optim.Optimizer]] = None,
        critic_optimizer_fn: Optional[Creates[torch.optim.Optimizer]] = None,
        critic_loss_fn: Optional[Creates[LossFn]] = None,
        stack_size: int = 1,
        replay_buffer: Optional[Creates[BaseReplayBuffer]] = None,
        discount_rate: float = 0.99,
        n_step: int = 1,
        grad_clip: Optional[float] = None,
        reward_clip: Optional[float] = None,
        soft_update_fraction: float = 0.005,
        batch_size: int = 64,
        log_frequency: int = 100,
        update_frequency: int = 1,
        action_noise: float = 0,
        min_replay_history: int = 1000,
        device="cpu",
        id=0,
    ):
        """
        Args:
            observation_space (gym.spaces.Box): Observation space for the agent.
            action_space (gym.spaces.Box): Action space for the agent.
            representation_net (torch.nn.Module): The network that encodes the
                observations that are then fed into the actor_net and critic_net. If
                None, defaults to :py:class:`~torch.nn.Identity`.
            actor_net (torch.nn.Module): The network that takes the encoded
                observations from representation_net and outputs the representations
                used to compute the actions (ie everything except the last layer).
            critic_net (torch.nn.Module): The network that takes two inputs: the
                encoded observations from representation_net and actions. It outputs
                the representations used to compute the values of the actions (ie
                everything except the last layer).
            init_fn (InitializationFn): Initializes the weights of agent networks using
                create_init_weights_fn.
            actor_optimizer_fn (torch.optim.Optimizer): A function that takes in the list of
                parameters of the actor returns the optimizer for the actor. If None,
                defaults to :py:class:`~torch.optim.Adam`.
            critic_optimizer_fn (torch.optim.Optimizer): A function that takes in the list of
                parameters of the critic returns the optimizer for the critic. If None,
                defaults to :py:class:`~torch.optim.Adam`.
            critic_loss_fn (LossFn): The loss function used to optimize the critic. If
                None, defaults to :py:class:`~torch.nn.MSELoss`.
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
            log_frequency (int): How often to log the agent's metrics.
            update_frequency (int): How frequently to update the agent. A value of 1
                means the agent will be updated every time update is called.
            action_noise (float): The standard deviation for the noise added to the
                action taken by the agent during training.
            min_replay_history (int): How many observations to fill the replay buffer
                with before starting to learn.
            device: Device on which all computations should be run.
            id: Agent identifier.
        """
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
            n_critics=1,
            stack_size=stack_size,
            replay_buffer=replay_buffer,
            discount_rate=discount_rate,
            n_step=n_step,
            grad_clip=grad_clip,
            reward_clip=reward_clip,
            soft_update_fraction=soft_update_fraction,
            batch_size=batch_size,
            log_frequency=log_frequency,
            update_frequency=update_frequency,
            policy_update_frequency=1,
            action_noise=action_noise,
            target_noise=0.0,
            target_noise_clip=0.0,
            min_replay_history=min_replay_history,
            device=device,
            id=id,
        )
