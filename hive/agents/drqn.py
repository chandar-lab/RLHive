import copy
import os

import numpy as np
import torch

from hive.agents.agent import Agent
from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.qnet_heads import DRQNNetwork
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
from hive.agents.dqn import DQNAgent
from hive.utils.utils import LossFn, OptimizerFn, create_folder, seeder


class DRQNAgent(DQNAgent):
    """An agent implementing the DRQN algorithm. Uses an epsilon greedy
    exploration policy
    """

    def __init__(
        self,
        representation_net: FunctionApproximator,
        obs_dim,
        act_dim: int,
        stack_size: int = 1,
        id: int = 0,
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
        store_hidden: bool = True,
        burn_frames: int = 0,
    ):
        super().__init__(
            representation_net,
            obs_dim,
            act_dim,
            stack_size,
            id,
            optimizer_fn,
            loss_fn,
            init_fn,
            replay_buffer,
            discount_rate,
            n_step,
            grad_clip,
            reward_clip,
            update_period_schedule,
            target_net_soft_update,
            target_net_update_fraction,
            target_net_update_schedule,
            epsilon_schedule,
            test_epsilon,
            min_replay_history,
            batch_size,
            device,
            logger,
            log_frequency,
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
        network = representation_net(self._obs_dim)
        network_output_dim = np.prod(calculate_output_dim(network, self._obs_dim)[0])
        self._qnet = DRQNNetwork(network, network_output_dim, self._act_dim).to(
            self._device
        )

        self._qnet.apply(self._init_fn)
        self._target_qnet = copy.deepcopy(self._qnet).requires_grad_(False)
        self._hidden_state = network.init_hidden(batch_size=1, device=self._device)
        self._rnn_type = network._rnn_type
        print(self._rnn_type)

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

        if self._rnn_type == "lstm" and self._store_hidden == True:
            preprocessed_update_info = {
                "observation": update_info["observation"],
                "action": update_info["action"],
                "reward": update_info["reward"],
                "done": update_info["done"],
                "hidden_state": self._prev_hidden_state,
                "cell_state": self._prev_cell_state,
            }

        elif self._rnn_type == "gru" and self._store_hidden == True:
            preprocessed_update_info = {
                "observation": update_info["observation"],
                "action": update_info["action"],
                "reward": update_info["reward"],
                "done": update_info["done"],
                "hidden_state": self._prev_hidden_state,
            }

        else:
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
            batch[key] = torch.tensor(batch[key], device=self._device)
        return (batch["observation"],), (batch["next_observation"],), batch

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

        if self._rnn_type == "lstm" and self._store_hidden == True:
            self._prev_hidden_state = self._hidden_state[0].detach().cpu().numpy()
            self._prev_cell_state = self._hidden_state[1].detach().cpu().numpy()

        elif self._rnn_type == "gru" and self._store_hidden == True:
            self._prev_hidden_state = self._hidden_state[0].detach().cpu().numpy()

        qvals, self._hidden_state = self._qnet(observation, self._hidden_state)
        if self._rng.random() < epsilon:
            action = self._rng.integers(self._act_dim)
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
        Updates the DQN agent.

        Args:
            update_info: dictionary containing all the necessary information to
                update the agent. Should contain a full transition, with keys for
                "observation", "action", "reward", and "done".
        """
        if update_info["done"]:
            self._state["episode_start"] = True
            self._hidden_state = self._qnet.base_network.init_hidden(
                batch_size=1, device=self._device
            )

        if not self._training:
            return

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

            if self._rnn_type == "lstm" and self._store_hidden == True:
                print("lstm and store hidden")
                hidden_state = (
                    torch.tensor(
                        batch["hidden_state"][:, 0].squeeze(1).squeeze(1).unsqueeze(0),
                        device=self._device,
                    ).float(),
                    torch.tensor(
                        batch["cell_state"][:, 0].squeeze(1).squeeze(1).unsqueeze(0),
                        device=self._device,
                    ).float(),
                )

                target_hidden_state = (
                    torch.tensor(
                        batch["next_hidden_state"][:, 0]
                        .squeeze(1)
                        .squeeze(1)
                        .unsqueeze(0),
                        device=self._device,
                    ).float(),
                    torch.tensor(
                        batch["next_cell_state"][:, 0]
                        .squeeze(1)
                        .squeeze(1)
                        .unsqueeze(0),
                        device=self._device,
                    ).float(),
                )

            elif self._rnn_type == "gru" and self._store_hidden == True:

                hidden_state = torch.tensor(
                    batch["hidden_state"][:, 0].squeeze(1).squeeze(1).unsqueeze(0),
                    device=self._device,
                ).float()

                target_hidden_state = torch.tensor(
                    batch["next_hidden_state"][:, 0].squeeze(1).squeeze(1).unsqueeze(0),
                    device=self._device,
                ).float()

            else:

                hidden_state = self._qnet.base_network.init_hidden(
                    batch_size=self._batch_size, device=self._device
                )

                target_hidden_state = self._target_qnet.base_network.init_hidden(
                    batch_size=self._batch_size, device=self._device
                )

            # Compute predicted Q values
            self._optimizer.zero_grad()
            pred_qvals, hidden_state = self._qnet(*current_state_inputs, hidden_state)
            pred_qvals = pred_qvals.view(
                self._batch_size, self._replay_buffer._max_seq_len, -1
            )
            actions = batch["action"].long()
            pred_qvals = torch.gather(pred_qvals, -1, actions.unsqueeze(-1)).squeeze(-1)

            # Compute 1-step Q targets
            next_qvals, target_hidden_state = self._target_qnet(
                *next_state_inputs, target_hidden_state
            )
            next_qvals = next_qvals.view(
                self._batch_size, self._replay_buffer._max_seq_len, -1
            )
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
