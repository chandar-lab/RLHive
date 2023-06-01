import copy

import gymnasium as gym
import numpy as np
import torch

from hive.agents.dqn import DQNAgent
from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.sequence_models import DRQNNetwork, SequenceFn, SequenceModel
from hive.agents.qnets.utils import (
    InitializationFn,
    calculate_output_dim,
    create_init_weights_fn,
    apply_to_tensor,
)
from hive.replays.recurrent_replay import RecurrentReplayBuffer
from hive.utils.loggers import Logger, NullLogger
from hive.utils.schedule import (
    LinearSchedule,
    PeriodicSchedule,
    Schedule,
    SwitchSchedule,
)
from hive.utils.utils import LossFn, OptimizerFn, seeder


class DRQNAgent(DQNAgent):
    """An agent implementing the DRQN algorithm. Uses an epsilon greedy
    exploration policy
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        representation_net: FunctionApproximator,
        sequence_fn: SequenceFn,
        id=0,
        optimizer_fn: OptimizerFn = None,
        loss_fn: LossFn = None,
        init_fn: InitializationFn = None,
        replay_buffer: RecurrentReplayBuffer = None,
        max_seq_len: int = 1,
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
        store_hidden: bool = True,
        burn_frames: int = 0,
        **kwargs,
    ):
        """
        Args:
            observation_space (gym.spaces.Box): Observation space for the agent.
            action_space (gym.spaces.Discrete): Action space for the agent.
            representation_net (SequenceFunctionApproximator): A network that outputs the
                representations that will be used to compute Q-values (e.g.
                everything except the final layer of the DRQN), as well as the
                hidden states of the recurrent component. The structure should be
                similar to ConvRNNNetwork, i.e., it should have a current module
                component placed between the convolutional layers and MLP layers.
                It should also define a method that initializes the hidden state
                of the recurrent module if the computation requires hidden states
                as input/output.
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
                :py:class:`~hive.replays.recurrent_replay.RecurrentReplayBuffer`.
            max_seq_len (int): The number of consecutive transitions in a sequence.
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
        self._max_seq_len = max_seq_len
        super(DQNAgent, self).__init__(
            observation_space=observation_space, action_space=action_space, id=id
        )
        self._state_size = (
            self._observation_space.shape[0],
            *self._observation_space.shape[1:],
        )
        self._init_fn = create_init_weights_fn(init_fn)
        self._device = torch.device("cpu" if not torch.cuda.is_available() else device)
        self.create_q_networks(representation_net, sequence_fn)
        if optimizer_fn is None:
            optimizer_fn = torch.optim.Adam
        self._optimizer = optimizer_fn(self._qnet.parameters())
        self._rng = np.random.default_rng(seed=seeder.get_new_seed("agent"))
        hidden_spec = self._qnet.get_hidden_spec()

        if not store_hidden or hidden_spec is None:
            store_hidden = False
            self._hidden_replay_spec = None
            self._hidden_batch_spec = None
        else:
            self._hidden_replay_spec = {key: hidden_spec[key][0] for key in hidden_spec}
            self._hidden_batch_spec = {key: hidden_spec[key][1] for key in hidden_spec}
        if replay_buffer is None:
            replay_buffer = RecurrentReplayBuffer
        self._replay_buffer = replay_buffer(
            observation_shape=self._observation_space.shape,
            observation_dtype=self._observation_space.dtype,
            action_shape=self._action_space.shape,
            action_dtype=self._action_space.dtype,
            max_seq_len=max_seq_len,
            hidden_spec=self._hidden_replay_spec,
        )
        self._discount_rate = discount_rate**n_step
        self._grad_clip = grad_clip
        self._reward_clip = reward_clip
        self._target_net_soft_update = target_net_soft_update
        self._target_net_update_fraction = target_net_update_fraction
        if loss_fn is None:
            loss_fn = torch.nn.SmoothL1Loss
        self._loss_fn = loss_fn(reduction="none")
        self._batch_size = batch_size
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

        self._training = False
        self._store_hidden = store_hidden
        self._burn_frames = burn_frames

    def create_q_networks(self, representation_net, sequence_fn):
        """Creates the Q-network and target Q-network.

        Args:
            representation_net: A network that outputs the representations that will
                be used to compute Q-values (e.g. everything except the final layer
                of the DRQN).
        """
        network = SequenceModel(
            self._state_size, representation_net(self._state_size), sequence_fn
        )
        network_output_dim = np.prod(
            calculate_output_dim(network, (1,) + self._state_size)[0]
        )
        self._qnet = DRQNNetwork(network, network_output_dim, self._action_space.n).to(
            self._device
        )

        self._qnet.apply(self._init_fn)
        self._target_qnet = copy.deepcopy(self._qnet).requires_grad_(False)

    def preprocess_observation(self, observation, agent_traj_state):
        # Reset hidden state if it is episode beginning.
        if agent_traj_state is None:
            hidden_state = self._qnet.init_hidden(batch_size=1)
        else:
            hidden_state = agent_traj_state["hidden_state"]

        state = torch.tensor(
            np.expand_dims(observation, axis=(0, 1)), device=self._device
        ).float()
        return state, hidden_state

    def preprocess_update_info(self, update_info, hidden_state):
        """Preprocesses the :obj:`update_info` before it goes into the replay buffer.
        Clips the reward in update_info.
        Args:
            update_info: Contains the information from the current timestep that the
                agent should use to update itself.
        """
        preprocessed_update_info = super().preprocess_update_info(update_info)

        if self._store_hidden:
            preprocessed_update_info.update(
                apply_to_tensor(hidden_state, lambda x: x.detach().cpu().numpy())
            )

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

        if self._store_hidden:
            for key in self._hidden_replay_spec:
                if self._hidden_batch_spec[key] >= 0:
                    # Replay batches on the first dimension, network expects
                    # batch on different dimension
                    batch[key] = torch.cat(
                        list(batch[key]), dim=self._hidden_batch_spec[key]
                    )
                    batch[f"next_{key}"] = torch.cat(
                        list(batch[f"next_{key}"]), dim=self._hidden_batch_spec[key]
                    )

            return (
                (
                    batch["observation"],
                    {key: batch[key] for key in self._hidden_replay_spec},
                ),
                (
                    batch["next_observation"],
                    {key: batch[f"next_{key}"] for key in self._hidden_replay_spec},
                ),
                batch,
            )
        else:
            return (batch["observation"]), (batch["next_observation"]), batch

    @torch.no_grad()
    def act(self, observation, agent_traj_state=None):
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
        # Insert batch_size and sequence_len dimensions to observation
        state, hidden_state = self.preprocess_observation(observation, agent_traj_state)
        qvals, hidden_state = self._qnet(state, hidden_state)
        if self._rng.random() < epsilon:
            action = self._rng.integers(self._action_space.n)
        else:
            # Note: not explicitly handling the ties
            action = torch.argmax(qvals).item()

        if (
            self._training
            and self._logger.should_log(self._timescale)
            and agent_traj_state is None
        ):
            self._logger.log_scalar("train_qval", torch.max(qvals), self._timescale)
        return action, {"hidden_state": hidden_state}

    def update(self, update_info, agent_traj_state=None):
        """
        Updates the DRQN agent.

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
        self._replay_buffer.add(
            **self.preprocess_update_info(
                update_info, hidden_state=agent_traj_state["hidden_state"]
            )
        )

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

            # Compute predicted Q values
            self._optimizer.zero_grad()

            pred_qvals, _ = self._qnet(*current_state_inputs)
            pred_qvals = pred_qvals.view(self._batch_size, self._max_seq_len, -1)
            actions = batch["action"].long()
            pred_qvals = torch.gather(pred_qvals, -1, actions.unsqueeze(-1)).squeeze(-1)

            # Compute 1-step Q targets
            next_qvals, _ = self._target_qnet(*next_state_inputs)
            next_qvals = next_qvals.view(self._batch_size, self._max_seq_len, -1)
            next_qvals, _ = torch.max(next_qvals, dim=-1)

            q_targets = batch["reward"] + self._discount_rate * next_qvals * (
                1 - batch["terminated"]
            )

            if self._burn_frames > 0:
                interm_loss = self._loss_fn(pred_qvals, q_targets)
                mask = torch.zeros(
                    self._replay_buffer._max_seq_len,
                    device=self._device,
                    dtype=torch.float,
                )
                mask[self._burn_frames :] = 1.0
                mask = mask.view(1, -1)
                interm_loss *= mask
                loss = interm_loss.mean()

            else:
                loss = self._loss_fn(pred_qvals, q_targets).mean()

            if self._logger.should_log(self._timescale):
                self._logger.log_scalar("train_loss", loss, self._timescale)

            loss.backward()
            if self._grad_clip is not None:
                torch.nn.utils.clip_grad_value_(
                    self._qnet.parameters(), self._grad_clip
                )
            self._optimizer.step()

        # Update target network
        if self._target_net_update_schedule.update():
            self._update_target()
        return agent_traj_state
