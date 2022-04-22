import copy
import os

import numpy as np
import torch
from flax import core
from flax.training import checkpoints
import gin
import jax
import jax.numpy as jnp
import rlax
import optax
import time

from hive.agents.agent import Agent
from hive.agents_jax.qnets.base import FunctionApproximator
from hive.agents_jax.qnets.qnet_heads import JaxDQNNetwork
from hive.agents_jax.qnets.utils import (
    InitializationFn,
    calculate_output_dim,
    create_init_weights_fn,
)
from hive.replays import BaseReplayBuffer, CircularReplayBuffer
from hive.utils.loggers import Logger, NullLogger
from hive.utils.schedule import (
    LinearSchedule,
    PeriodicSchedule,
    Schedule,
    SwitchSchedule,
)
from hive.utils.utils import LossFn, OptimizerFn, create_folder

## TODO  I create a seeder here for Jax instead of what we have in utils
class jax_Seeder:
    """Class used to manage seeding in RLHive. It sets the seed for all the frameworks
    that RLHive currently uses. It also deterministically provides new seeds based on
    the global seed, in case any other objects in RLHive (such as the agents) need
    their own seed.
    """

    def __init__(self):
        self._seed = 0
        self._current_seed = 0

    def set_global_seed(self, seed):
        """This reduces some sources of randomness in experiments. To get reproducible
        results, you must run on the same machine and set the environment variable
        CUBLAS_WORKSPACE_CONFIG to ":4096:8" or ":16:8" before starting the experiment.

        Args:
            seed (int): Global seed.
        """
        self._seed = seed
        self._current_seed = seed
        # torch.manual_seed(self._seed)
        # random.seed(self._seed)
        # np.random.seed(self._seed)
        # torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)

    def get_new_seed(self):
        """Each time it is called, it increments the current_seed and returns it."""
        self._current_seed += 1
        return self._current_seed


class DQNAgent(Agent):
    """An agent implementing the DQN algorithm. Uses an epsilon greedy
    exploration policy
    """

    def __init__(
        self,
        representation_net: FunctionApproximator,
        obs_dim,
        act_dim: int,
        id=0,
        optimizer_fn: OptimizerFn = None,
        loss_fn: LossFn = None,
        init_fn: InitializationFn = None,
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
        seed=None,
        stack_size: int = 1,
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
            stack_size (int): The number of frames to stack to create an observation.
        """
        super().__init__(obs_dim=obs_dim, act_dim=act_dim, id=id)
        self._init_fn = create_init_weights_fn(init_fn)
        self.create_q_networks(representation_net)
        if optimizer_fn is None:
            optimizer_fn = optax.adam()
        self.optimizer = optimizer_fn
        self.optimizer_state = self.optimizer.init(self.online_params)

        # seed = int(time.time() * 1e6) if seed is None else seed
        self._rng = jax.random.PRNGKey(seed=jax_Seeder.get_new_seed())

        self._replay_buffer = replay_buffer
        if self._replay_buffer is None:
            self._replay_buffer = CircularReplayBuffer()
        self._discount_rate = discount_rate**n_step
        self._grad_clip = grad_clip
        self._reward_clip = reward_clip
        self._target_net_soft_update = target_net_soft_update
        self._target_net_update_fraction = target_net_update_fraction

        # def loss_fn(params, target):
        #     def q_online(state):
        #         return network_def.apply(params, state)
        #
        #     q_values = jax.vmap(q_online)(states).q_values
        #     q_values = jnp.squeeze(q_values)
        #     replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
        #     if loss_type == 'huber':
        #         return jnp.mean(jax.vmap(losses.huber_loss)(target, replay_chosen_q))
        #     return jnp.mean(jax.vmap(losses.mse_loss)(target, replay_chosen_q))

        if loss_fn is None:
            loss_fn = rlax.l2_loss
        self._loss_fn = loss_fn

        self._batch_size = batch_size
        self._logger = logger
        if self._logger is None:
            self._logger = NullLogger([])
        self._timescale = self.id
        self._logger.register_timescale(
            self._timescale, PeriodicSchedule(False, True, log_frequency)
        )
        self._update_period_schedule = update_period_schedule
        if self._update_period_schedule is None:
            self._update_period_schedule = PeriodicSchedule(False, True, 1)
        self._target_net_update_schedule = target_net_update_schedule
        if self._target_net_update_schedule is None:
            self._target_net_update_schedule = PeriodicSchedule(False, True, 10000)
        self._epsilon_schedule = epsilon_schedule
        if self._epsilon_schedule is None:
            self._epsilon_schedule = LinearSchedule(1, 0.1, 100000)
        self._test_epsilon = test_epsilon
        self._learn_schedule = SwitchSchedule(False, True, min_replay_history)

        self._state = {"episode_start": True}
        self._training = False
        self.stack_size = stack_size

        state_shape = obs_dim + (stack_size,)  ## check stack size
        self.sample_network_input = jnp.zeros(state_shape)
        self.batch_q_learning = jax.vmap(rlax.q_learning)

    def create_q_networks(self, representation_net):
        """Creates the Q-network and target Q-network.
        Args:
            representation_net: A network that outputs the representations that will
                be used to compute Q-values (e.g. everything except the final layer
                of the DQN).
        """
        network = representation_net(self._obs_dim)
        network_output_dim = jnp.prod(calculate_output_dim(network, self._obs_dim))
        self._qnet = JaxDQNNetwork(network, network_output_dim, self._act_dim)
        self._rng, rng = jax.random.split(self._rng)
        self.online_params = self._qnet.init(rng, x=self.sample_network_input)
        self._target_qnet_params = self.online_params

    def train(self):
        """Changes the agent to training mode."""
        super().train()
        self._qnet.train()

    def eval(self):
        """Changes the agent to evaluation mode."""
        super().eval()
        self._qnet.eval()

    def preprocess_update_info(self, update_info):
        """Preprocesses the :obj:`update_info` before it goes into the replay buffer.
        Clips the reward in update_info.
        Args:
            update_info: Contains the information from the current timestep that the
                agent should use to update itself.
        """
        if self._reward_clip is not None:
            update_info["reward"] = jnp.clip(
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
            (tuple):
                - (tuple) Inputs used to calculate current state values.
                - (tuple) Inputs used to calculate next state values
                - Preprocessed batch.
        """
        for key in batch:
            batch[key] = jax.device_put(batch[key])
        return (batch["observation"],), (batch["next_observation"],), batch

    @torch.no_grad()
    def act(self, observation):  ## it this batch?
        """Returns the action for the agent. If in training mode, follows an epsilon
        greedy policy. Otherwise, returns the action with the highest Q-value.
        Args:
            observation: The current observation.
        """

        # Determine and log the value of epsilon
        if self._training:
            if not self._learn_schedule.get_value():
                epsilon = 1.0
            else:
                epsilon = self._epsilon_schedule.update()
            if self._logger.update_step(self._timescale):
                self._logger.log_scalar("epsilon", epsilon, self._timescale)
        else:
            epsilon = self._test_epsilon

        def q_online(observation):
            return self._qnet.apply(self.online_params, observation)

        q_values = jax.vmap(q_online)(observation).q_values
        q_values = jnp.squeeze(q_values)

        if self._rng.random() < epsilon:
            action = self._rng.integers(self._act_dim)
        else:
            # Note: not explicitly handling the ties
            action = jnp.argmax(q_values)

        if (
            self._training
            and self._logger.should_log(self._timescale)
            and self._state["episode_start"]
        ):
            self._logger.log_scalar("train_qval", jnp.max(q_values), self._timescale)
            self._state["episode_start"] = False
        return action

    def update(self, update_info):
        """
        Updates the DQN agent.
        Args:
            update_info: dictionary containing all the necessary information to
                update the agent. Should contain a full transition, with keys for
                "observation", "action", "reward", and "done".
        """
        if update_info["done"]:
            self._state["episode_start"] = True

        if not self._training:
            return

        def loss_fn(batch):
            q_tm1 = self._qnet.apply(self.online_params, batch["observation"]).q_values
            q_target_t = self._qnet.apply(
                self._target_qnet_params, batch["next_observation"]
            ).q_values
            td_errors = self.batch_q_learning(
                q_tm1,
                batch["action"],
                batch["reward"],
                batch["observation"],
                self._discount_rate,
                q_target_t,
            )
            if self._grad_clip is not None:
                td_errors = rlax.clip_gradient(
                    td_errors, -self._grad_clip, self._grad_clip
                )
            else:
                td_errors = rlax.clip_gradient(td_errors)

            losses = rlax.l2_loss(td_errors)
            return losses

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

            batch_gard = jax.grad(loss_fn)(batch)
            updates, self.optimizer_state = self.optimizer.update(
                batch_gard, self.optimizer_state
            )
            self.online_params = optax.apply_updates(self.online_params, updates)
            self._target_qnet_params = self.online_params

    ### TODO work on def save(self, dname):

    ### TODO work on def load(self, dname):
