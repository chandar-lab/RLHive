import copy
import os

import gym
import numpy as np
import torch

from hive.agents.agent import Agent
from hive.agents.qnets.base import FunctionApproximator
from hive.agents.world_models.base import WorldModel
from hive.agents.qnets.qnet_heads import DQNNetwork
from hive.agents.qnets.utils import (
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
from hive.utils.utils import LossFn, OptimizerFn, create_folder, seeder


class NonlinearDynaQ(Agent):
    """An agent implementing the nonlinear Dyna-Q algorithm."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        dyna_model: WorldModel,
        value_representation_net: FunctionApproximator,
        stack_size: int = 1,
        id=0,
        model_optimizer_fn: OptimizerFn = None,
        value_optimizer_fn: OptimizerFn = None,
        observation_loss_fn: LossFn = None,
        reward_loss_fn: LossFn = None,
        done_loss_fn: LossFn = None,
        value_loss_fn: LossFn = None,
        init_fn: InitializationFn = None,
        learning_buffer: BaseReplayBuffer = None,
        planning_buffer: BaseReplayBuffer = None,
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
        learning_batch_size: int = 32,
        planning_batch_size: int = 32,
        num_learning_steps: int = 1,
        num_planning_steps: int = 1,
        device="cpu",
        logger: Logger = None,
        log_frequency: int = 100,
    ):
        """
        Args:
            observation_space (gym.spaces.Box): Observation space for the agent.
            action_space (gym.spaces.Discrete): Action space for the agent.
            dyna_model (WorldModel): A network that outputs the next observation, reward,
                and episode termination given a state and action.
            value_representation_net (FunctionApproximator): A network that outputs the
                representations that will be used to compute Q-values.
            stack_size: Number of observations stacked to create the observation fed to the
                general neural network.
            id: Agent identifier.
            model_optimizer_fn (OptimizerFn): A function that takes in a list of parameters
                to optimize and returns the optimizer for learning the model.
                If None, defaults to :py:class:`~torch.optim.Adam`.
            value_optimizer_fn (OptimizerFn): A function that takes in a list of parameters
                to optimize and returns the optimizer for learning the Q-values.
                If None, defaults to :py:class:`~torch.optim.Adam`.
            observation_loss_fn (LossFn): Loss function used by the agent for the observations.
                If None, defaults to :py:class:`~torch.nn.MSELoss`.
            reward_loss_fn (LossFn): Loss function used by the agent for the rewards.
                If None, defaults to :py:class:`~torch.nn.MSELoss`.
            done_loss_fn (LossFn): Loss function used by the agent for the episode termination.
                If None, defaults to :py:class:`~torch.nn.BCELoss`.
            value_loss_fn (LossFn): Loss function used by the agent for the Q-values.
                If None, defaults to :py:class:`~torch.nn.MSELoss`.
            init_fn (InitializationFn): Initializes the weights of general neural network
                using create_init_weights_fn.
            learning_buffer (BaseReplayBuffer): The replay buffer that the agent will
                push observations to and sample from during model learning. If None,
                defaults to
                :py:class:`~hive.replays.circular_replay.CircularReplayBuffer`.
            planning_buffer (BaseReplayBuffer): The replay buffer that the agent will
                push observations to and sample from during planning. If None,
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
            learning_batch_size (int): The size of the batch sampled from
                the learning replay buffer during learning.
            planning_batch_size (int): The size of the batch sampled from
                the planning replay buffer during learning.
            num_learning_steps (int): The number of times to learn the model
                per each training step.
            num_planning_steps (int): The number of times to do the planning
                per each training step.
            device: Device on which all computations should be run.
            logger (ScheduledLogger): Logger used to log agent's metrics.
            log_frequency (int): How often to log the agent's metrics.
        """
        super().__init__(
            observation_space=observation_space, action_space=action_space, id=id
        )

        self._state_size = (
            stack_size * self._observation_space.shape[0],
            *self._observation_space.shape[1:],
        )
        self._init_fn = create_init_weights_fn(init_fn)
        self._device = torch.device("cpu" if not torch.cuda.is_available() else device)
        self.create_networks(dyna_model, value_representation_net)
        if model_optimizer_fn is None:
            model_optimizer_fn = torch.optim.Adam
        if value_optimizer_fn is None:
            value_optimizer_fn = torch.optim.Adam
        self._model_optimizer = model_optimizer_fn(self._model_network.parameters())
        self._value_optimizer = value_optimizer_fn(self._value_network.parameters())
        if observation_loss_fn is None:
            observation_loss_fn = torch.nn.MSELoss
        if reward_loss_fn is None:
            reward_loss_fn = torch.nn.MSELoss
        if done_loss_fn is None:
            done_loss_fn = torch.nn.BCELoss
        if value_loss_fn is None:
            value_loss_fn = torch.nn.MSELoss
        self._observation_loss_fn = observation_loss_fn(reduction="none")
        self._reward_loss_fn = reward_loss_fn(reduction="none")
        self._done_loss_fn = done_loss_fn(reduction="none")
        self._value_loss_fn = value_loss_fn(reduction="none")
        if learning_buffer is None:
            learning_buffer = CircularReplayBuffer
        if planning_buffer is None:
            planning_buffer = CircularReplayBuffer
        self._learning_buffer = learning_buffer(
            observation_shape=self._observation_space.shape,
            observation_dtype=self._observation_space.dtype,
            action_shape=self._action_space.shape,
            action_dtype=self._action_space.dtype,
            gamma=discount_rate,
        )
        self._planning_buffer = planning_buffer(
            observation_shape=self._observation_space.shape,
            observation_dtype=self._observation_space.dtype,
            action_shape=self._action_space.shape,
            action_dtype=self._action_space.dtype,
            gamma=discount_rate,
        )
        self._discount_rate = discount_rate**n_step
        self._grad_clip = grad_clip
        self._reward_clip = reward_clip
        self._target_net_soft_update = target_net_soft_update
        self._target_net_update_fraction = target_net_update_fraction
        self._learning_batch_size = learning_batch_size
        self._planning_batch_size = planning_batch_size
        self._num_learning_steps = num_learning_steps
        self._num_planning_steps = num_planning_steps
        self._rng = np.random.default_rng(seed=seeder.get_new_seed())
        self._logger = logger
        if self._logger is None:
            self._logger = NullLogger([])
        self._timescale = self.id
        self._logger.register_timescale(
            self._timescale, PeriodicSchedule(False, True, log_frequency)
        )
        if update_period_schedule is None:
            self._update_period_schedule = PeriodicSchedule(False, True, 1)
        else:
            self._update_period_schedule = update_period_schedule()

        if target_net_update_schedule is None:
            self._target_net_update_schedule = PeriodicSchedule(False, True, 10000)
        else:
            self._target_net_update_schedule = target_net_update_schedule()

        if epsilon_schedule is None:
            self._epsilon_schedule = LinearSchedule(1, 0.1, 100000)
        else:
            self._epsilon_schedule = epsilon_schedule()

        self._test_epsilon = test_epsilon
        self._learn_schedule = SwitchSchedule(False, True, min_replay_history)
        self._state = {"episode_start": True}
        self._training = False

    def create_networks(self, dyna_model, value_representation_net):
        """Creates the Q-network and target Q-network.

        Args:
            dyna_model: The network that will be used to compute model's estimations.
            value_representation_net: A network that outputs the representations that will
                    be used to compute Q-values (e.g. everything except the final layer).
        """
        # Model
        self._model_network = dyna_model(self._state_size, self._action_space.n).to(
            self._device
        )
        self._model_network.apply(self._init_fn)
        self._target_model_network = copy.deepcopy(self._model_network).requires_grad_(
            False
        )

        # Value
        value_network_repr = value_representation_net(self._state_size)
        network_output_dim = np.prod(
            calculate_output_dim(value_network_repr, self._state_size)
        )
        self._value_network = DQNNetwork(
            value_network_repr, network_output_dim, self._action_space.n
        ).to(self._device)
        self._value_network.apply(self._init_fn)
        self._target_value_network = copy.deepcopy(self._value_network).requires_grad_(
            False
        )

    def train(self):
        """Changes the agent to training mode."""
        super().train()
        self._value_network.train()
        self._target_value_network.train()
        self._model_network.train()
        self._target_model_network.train()

    def eval(self):
        """Changes the agent to evaluation mode."""
        super().eval()
        self._value_network.eval()
        self._target_value_network.eval()
        self._model_network.eval()
        self._target_model_network.eval()

    def preprocess_update_info(self, update_info):
        """Preprocesses the :obj:`update_info` before it goes into the replay buffers.
        Clips the reward in update_info.

        Args:
            update_info: Contains the information from the current timestep that the
                agent should use to update itself.
        """
        if self._reward_clip is not None:
            update_info["reward"] = np.clip(
                update_info["reward"], -self._reward_clip, self._reward_clip
            )
        preprocessed_learning_update_info = {
            "observation": update_info["observation"],
            "action": update_info["action"],
            "reward": update_info["reward"],
            "done": update_info["done"],
        }

        if isinstance(update_info["action"], int):
            planning_action_sample_size = np.array([update_info["action"]]).shape
        else:
            planning_action_sample_size = np.array(update_info["action"]).shape
        # TODO remove reward and done from the planning buffer,
        #  since they are not needed
        preprocessed_planning_update_info = {
            "observation": update_info["observation"],
            "action": np.random.randint(
                self._action_space.n, size=planning_action_sample_size
            ),
            "reward": update_info["reward"],
            "done": update_info["done"],
        }

        if "agent_id" in update_info:
            preprocessed_learning_update_info["agent_id"] = int(update_info["agent_id"])
            preprocessed_planning_update_info["agent_id"] = int(update_info["agent_id"])

        return preprocessed_learning_update_info, preprocessed_planning_update_info

    def preprocess_update_batch(self, batch):
        """Preprocess the batch sampled from the replay buffers.

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

        # Sample action. With epsilon probability choose random action,
        # otherwise select the action with the highest q-value.
        observation = torch.tensor(
            np.expand_dims(observation, axis=0), device=self._device
        ).float()
        qvals = self._value_network(observation)
        if self._rng.random() < epsilon:
            action = self._rng.integers(self._action_space.n)
        else:
            # Note: not explicitly handling the ties
            action = torch.argmax(qvals).item()

        if (
            self._training
            and self._logger.should_log(self._timescale)
            and self._state["episode_start"]
        ):
            self._logger.log_scalar("train_qval", torch.max(qvals), self._timescale)
            self._state["episode_start"] = False
        return action

    def update(self, update_info):
        """
        Updates the Dyna-Q agent.

        Args:
            update_info: dictionary containing all the necessary information to
                update the agent. Should contain a full transition, with keys for
                "observation", "action", "reward", and "done".
        """
        if update_info["done"]:
            self._state["episode_start"] = True

        if not self._training:
            return

        (
            preprocessed_learning_update_info,
            preprocessed_planning_update_info,
        ) = self.preprocess_update_info(update_info)
        self._learning_buffer.add(**preprocessed_learning_update_info)
        self._planning_buffer.add(**preprocessed_planning_update_info)

        if (
            self._learn_schedule.update()
            and self._learning_buffer.size() > 0
            and self._update_period_schedule.update()
        ):
            # Model Learning
            model_losses = []
            for _ in range(self._num_learning_steps):
                batch = self._learning_buffer.sample(
                    batch_size=self._learning_batch_size
                )
                batch = self.preprocess_update_batch(batch)

                self._model_optimizer.zero_grad()
                next_obs_pred, reward_pred, done_pred = self._model_network(
                    batch["observation"], batch["action"]
                )
                obs_loss = (
                    self._observation_loss_fn(
                        next_obs_pred, batch["next_observation"]
                    ).sum(1)
                    * (1 - batch["done"])
                ).mean()
                reward_loss = self._reward_loss_fn(reward_pred, batch["reward"]).mean()
                # TODO Fix this .float()
                done_loss = self._done_loss_fn(done_pred, batch["done"].float()).mean()

                model_loss = obs_loss + reward_loss + done_loss
                model_losses.append(model_loss.item())
                model_loss.backward()
                if self._grad_clip is not None:
                    torch.nn.utils.clip_grad_value_(
                        self._model_network.parameters(), self._grad_clip
                    )
                self._model_optimizer.step()

            # Value Learning (Planning)
            planning_losses = []
            for _ in range(self._num_planning_steps):
                batch = self._planning_buffer.sample(
                    batch_size=self._planning_batch_size
                )
                batch = self.preprocess_update_batch(batch)

                self._value_optimizer.zero_grad()
                with torch.no_grad():
                    next_obs_pred, reward_pred, done_pred = self._model_network(
                        batch["observation"], batch["action"]
                    )
                pred_qvals = self._value_network(batch["observation"])
                actions = batch["action"].long()
                pred_qvals = pred_qvals[torch.arange(pred_qvals.size(0)), actions]

                next_qvals = self._target_value_network(next_obs_pred)
                next_qvals, _ = torch.max(next_qvals, dim=1)

                q_targets = reward_pred + self._discount_rate * next_qvals * (
                    1 - done_pred
                )

                planning_loss = self._value_loss_fn(pred_qvals, q_targets).mean()
                planning_losses.append(planning_loss.item())
                planning_loss.backward()
                if self._grad_clip is not None:
                    torch.nn.utils.clip_grad_value_(
                        self._value_network.parameters(), self._grad_clip
                    )
                self._value_optimizer.step()

            if self._logger.should_log(self._timescale):
                self._logger.log_scalar(
                    "average_model_loss", np.mean(model_losses), self._timescale
                )
                self._logger.log_scalar(
                    "average_planning_loss", np.mean(planning_losses), self._timescale
                )

        # Update target network
        if self._target_net_update_schedule.update():
            self._update_target()

    def _update_target(self):
        """Update the target network."""
        if self._target_net_soft_update:
            target_params = [
                self._target_value_network.state_dict(),
                self._target_model_network,
            ]
            current_params = [self._value_network.state_dict(), self._model_network]
            for i in range(len(target_params)):
                for key in list(target_params[i].keys()):
                    target_params[i][key] = (
                        1 - self._target_net_update_fraction
                    ) * target_params[i][
                        key
                    ] + self._target_net_update_fraction * current_params[
                        i
                    ][
                        key
                    ]
            self._target_value_network.load_state_dict(target_params[0])
            self._target_model_network.load_state_dict(target_params[1])
        else:
            self._target_value_network.load_state_dict(self._value_network.state_dict())
            self._target_model_network.load_state_dict(self._model_network.state_dict())

    def save(self, dname):
        torch.save(
            {
                "value_network": self._value_network.state_dict(),
                "target_value_network": self._target_value_network.state_dict(),
                "model_network": self._model_network.state_dict(),
                "target_model_network": self._target_model_network.state_dict(),
                "value_optimizer": self._value_optimizer.state_dict(),
                "model_optimizer": self._model_optimizer.state_dict(),
                "learn_schedule": self._learn_schedule,
                "epsilon_schedule": self._epsilon_schedule,
                "target_net_update_schedule": self._target_net_update_schedule,
                "rng": self._rng,
            },
            os.path.join(dname, "agent.pt"),
        )
        replay_dir = os.path.join(dname, "learning_buffer")
        create_folder(replay_dir)
        self._learning_buffer.save(replay_dir)
        replay_dir = os.path.join(dname, "planning_buffer")
        create_folder(replay_dir)
        self._planning_buffer.save(replay_dir)

    def load(self, dname):
        checkpoint = torch.load(os.path.join(dname, "agent.pt"))
        self._value_network.load_state_dict(checkpoint["value_network"])
        self._target_value_network.load_state_dict(checkpoint["target_value_network"])
        self._model_network.load_state_dict(checkpoint["model_network"])
        self._target_model_network.load_state_dict(checkpoint["target_model_network"])
        self._value_optimizer.load_state_dict(checkpoint["value_optimizer"])
        self._model_optimizer.load_state_dict(checkpoint["model_optimizer"])
        self._learn_schedule = checkpoint["learn_schedule"]
        self._epsilon_schedule = checkpoint["epsilon_schedule"]
        self._target_net_update_schedule = checkpoint["target_net_update_schedule"]
        self._rng = checkpoint["rng"]
        self._learning_buffer.load(os.path.join(dname, "learning_buffer"))
        self._planning_buffer.load(os.path.join(dname, "planning_buffer"))
