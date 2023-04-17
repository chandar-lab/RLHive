import copy
import os
from functools import partial

import gymnasium as gym
import numpy as np
import torch

from hive.agents.agent import Agent
from hive.agents.dqn import DQNAgent
from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.qnet_heads import DRQNNetwork
from hive.agents.qnets.utils import (
    InitializationFn,
    calculate_output_dim,
    create_init_weights_fn,
)
from hive.replays import BaseReplayBuffer, CircularReplayBuffer
from hive.replays.recurrent_replay import RecurrentReplayBuffer
from hive.utils.loggers import Logger, NullLogger
from hive.utils.schedule import (
    LinearSchedule,
    PeriodicSchedule,
    Schedule,
    SwitchSchedule,
)
from hive.utils.utils import LossFn, OptimizerFn, create_folder, seeder


class DRQNAgent(DQNAgent):
    """An agent implementing the DRQN algorithm. Uses an epsilon greedy
    exploration policy
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        representation_net: FunctionApproximator,
        id=0,
        optimizer_fn: OptimizerFn = None,
        loss_fn: LossFn = None,
        init_fn: InitializationFn = None,
        replay_buffer: BaseReplayBuffer = None,
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
            representation_net (FunctionApproximator): A network that outputs the
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
        if replay_buffer is None:
            replay_buffer = RecurrentReplayBuffer
        replay_buffer = partial(
            replay_buffer, max_seq_len=max_seq_len, store_hidden=store_hidden
        )
        self._max_seq_len = max_seq_len
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            representation_net=representation_net,
            id=id,
            optimizer_fn=optimizer_fn,
            loss_fn=loss_fn,
            init_fn=init_fn,
            replay_buffer=replay_buffer,
            discount_rate=discount_rate,
            n_step=n_step,
            grad_clip=grad_clip,
            reward_clip=reward_clip,
            update_period_schedule=update_period_schedule,
            target_net_soft_update=target_net_soft_update,
            target_net_update_fraction=target_net_update_fraction,
            target_net_update_schedule=target_net_update_schedule,
            epsilon_schedule=epsilon_schedule,
            test_epsilon=test_epsilon,
            min_replay_history=min_replay_history,
            batch_size=batch_size,
            device=device,
            logger=logger,
            log_frequency=log_frequency,
        )
        self._store_hidden = store_hidden
        self._burn_frames = burn_frames

    def create_q_networks(self, representation_net):
        """Creates the Q-network and target Q-network.

        Args:
            representation_net: A network that outputs the representations that will
                be used to compute Q-values (e.g. everything except the final layer
                of the DRQN).
        """
        network = representation_net(self._state_size)

        if isinstance(network.rnn.core, torch.nn.LSTM):
            self._rnn_type = "lstm"
        elif isinstance(network.rnn.core, torch.nn.GRU):
            self._rnn_type = "gru"
        else:
            raise ValueError(
                f"rnn_type is wrong. Expected either lstm or gru,"
                f"received {network.rnn.core}."
            )

        network_output_dim = np.prod(
            calculate_output_dim(network, (1,) + self._state_size)[0]
        )
        self._qnet = DRQNNetwork(network, network_output_dim, self._action_space.n).to(
            self._device
        )
        self._qnet.update_rnn_device()

        self._qnet.apply(self._init_fn)
        self._target_qnet = copy.deepcopy(self._qnet).requires_grad_(False)

    def preprocess_update_info(self, update_info):
        """Preprocesses the :obj:`update_info` before it goes into the replay buffer.
        Clips the reward in update_info.
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
            "done": update_info["terminated"] or update_info["truncated"],
        }

        if self._store_hidden == True:
            preprocessed_update_info.update(
                self.unpack_hidden_state(update_info["hidden_state"])
            )

        if "agent_id" in update_info:
            preprocessed_update_info["agent_id"] = int(update_info["agent_id"])

        return preprocessed_update_info

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

        # Reset hidden state if it is episode beginning.
        if agent_traj_state is None:
            hidden_state = self._qnet.init_hidden(batch_size=1)
        else:
            hidden_state = agent_traj_state["hidden_state"]

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
        observation = torch.tensor(
            np.expand_dims(observation, axis=(0, 1)), device=self._device
        ).float()
        qvals, hidden_state = self._qnet(observation, hidden_state)
        if self._rng.random() < epsilon:
            action = self._rng.integers(self._action_space.n)
        else:
            # Note: not explicitly handling the ties
            action = torch.argmax(qvals).item()
        if agent_traj_state is None:
            agent_traj_state = {}
            if self._training and self._logger.should_log(self._timescale):
                self._logger.log_scalar("train_qval", torch.max(qvals), self._timescale)

        agent_traj_state["hidden_state"] = hidden_state
        return action, agent_traj_state

    def update(self, update_info, agent_traj_state=None):
        """
        Updates the DRQN agent.

        Args:
            update_info: dictionary containing all the necessary information
                from the environment to update the agent. Should contain a full
                transition, with keys for "observation", "action", "reward",
                "next_observation", and "done".
            agent_traj_state: Contains necessary state information for the agent
                to process current trajectory. This should be updated and returned.
        Returns:
            - action
            - agent trajectory state
        """
        if not self._training:
            return

        # Add the most recent transition to the replay buffer.
        update_info.update(agent_traj_state)
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

            hidden_state, target_hidden_state = self.get_hidden_state(batch)

            # Compute predicted Q values
            self._optimizer.zero_grad()
            pred_qvals, _ = self._qnet(*current_state_inputs, hidden_state)
            pred_qvals = pred_qvals.view(self._batch_size, self._max_seq_len, -1)
            actions = batch["action"].long()
            pred_qvals = torch.gather(pred_qvals, -1, actions.unsqueeze(-1)).squeeze(-1)

            # Compute 1-step Q targets
            next_qvals, _ = self._target_qnet(*next_state_inputs, target_hidden_state)
            next_qvals = next_qvals.view(self._batch_size, self._max_seq_len, -1)
            next_qvals, _ = torch.max(next_qvals, dim=-1)

            q_targets = batch["reward"] + self._discount_rate * next_qvals * (
                1 - batch["done"]
            )

            if self._burn_frames > 0:
                interm_loss = self._loss_fn(pred_qvals, q_targets)
                mask = torch.zeros(
                    self._replay_buffer._max_seq_len,
                    device=self._device,
                    dtype=torch.float,
                )
                mask[self._burn_frames :] = 1.0
                mask = mask.unsqueeze(0).repeat(len(batch["reward"]), 1)
                mask = mask & batch["mask"]
                interm_loss *= mask
                loss = interm_loss.sum() / mask.sum()

            else:
                interm_loss = self._loss_fn(pred_qvals, q_targets)
                interm_loss *= batch["mask"]
                loss = interm_loss.sum() / batch["mask"].sum()

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

    def unpack_hidden_state(self, hidden_state):
        if self._rnn_type == "lstm":
            hidden_state = {
                "hidden_state": hidden_state[0].detach().cpu().numpy(),
                "cell_state": hidden_state[1].detach().cpu().numpy(),
            }

        elif self._rnn_type == "gru":
            hidden_state = {
                "hidden_state": hidden_state[0].detach().cpu().numpy(),
            }
        else:
            raise ValueError(
                f"rnn_type is wrong. Expected either lstm or gru,"
                f"received {self._rnn_type}."
            )

        return hidden_state

    def get_hidden_state(self, batch):
        if self._store_hidden == True:
            hidden_state = (
                torch.tensor(
                    batch["hidden_state"][:, 0].squeeze(1).squeeze(1).unsqueeze(0),
                    device=self._device,
                ).float(),
            )

            target_hidden_state = (
                torch.tensor(
                    batch["next_hidden_state"][:, 0].squeeze(1).squeeze(1).unsqueeze(0),
                    device=self._device,
                ).float(),
            )

            if self._rnn_type == "lstm":
                hidden_state += (
                    torch.tensor(
                        batch["cell_state"][:, 0].squeeze(1).squeeze(1).unsqueeze(0),
                        device=self._device,
                    ).float(),
                )

                target_hidden_state += (
                    torch.tensor(
                        batch["next_cell_state"][:, 0]
                        .squeeze(1)
                        .squeeze(1)
                        .unsqueeze(0),
                        device=self._device,
                    ).float(),
                )
        else:
            hidden_state = self._qnet.init_hidden(
                batch_size=self._batch_size,
            )
            target_hidden_state = self._target_qnet.init_hidden(
                batch_size=self._batch_size,
            )

        return hidden_state, target_hidden_state
