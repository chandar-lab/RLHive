import os
import copy
import numpy as np
import torch

from hive.replays import CircularReplayBuffer, get_replay
from hive.utils.logging import NullLogger, get_logger
from hive.utils.utils import create_folder, get_optimizer_fn
from hive.utils.schedule import (
    PeriodicSchedule,
    LinearSchedule,
    SwitchSchedule,
    get_schedule,
)
from hive.agents.agent import Agent
from hive.agents.rainbow import RainbowDQNAgent
from hive.agents.qnets import get_qnet


class HanabiRainbowAgent(RainbowDQNAgent):
    """An agent implementing the DQN algorithm. Uses an epsilon greedy
    exploration policy
    """

    def __init__(
        self,
        qnet,
        obs_dim,
        act_dim,
        v_min=0,
        v_max=200,
        atoms=51,
        optimizer_fn=None,
        id=0,
        replay_buffer=None,
        discount_rate=0.99,
        grad_clip=None,
        target_net_soft_update=False,
        target_net_update_fraction=0.05,
        target_net_update_schedule=None,
        epsilon_schedule=None,
        learn_schedule=None,
        seed=42,
        batch_size=32,
        device="cpu",
        logger=None,
        log_frequency=100,
        double=True,
        distributional=False,
        epsilon_on=True,
    ):
        """
        Args:

        """
        super().__init__(
            qnet,
            obs_dim,
            act_dim,
            v_min=v_min,
            v_max=v_max,
            atoms=atoms,
            optimizer_fn=optimizer_fn,
            id=id,
            replay_buffer=replay_buffer,
            discount_rate=discount_rate,
            grad_clip=grad_clip,
            target_net_soft_update=target_net_soft_update,
            target_net_update_fraction=target_net_update_fraction,
            target_net_update_schedule=target_net_update_schedule,
            epsilon_schedule=epsilon_schedule,
            learn_schedule=learn_schedule,
            seed=seed,
            batch_size=batch_size,
            device=device,
            logger=logger,
            log_frequency=log_frequency,
            double=double,
            distributional=distributional,
            epsilon_on=epsilon_on,
        )

    def projection_distribution(self, batch):
        batch_obs = batch["observation"]
        batch_action = batch["action"].long()
        batch_next_obs = batch["next_observation"]
        batch_reward = batch["reward"].reshape(-1, 1).to(self._device)
        batch_not_done = 1 - batch["done"].reshape(-1, 1).to(self._device)

        with torch.no_grad():
            next_action = self._target_qnet(batch_next_obs).argmax(1)
            next_dist = self._target_qnet.dist(batch_next_obs)
            next_dist = next_dist[range(self._batch_size), next_action]

            t_z = batch_reward + batch_not_done * self._discount_rate * self._supports
            t_z = t_z.clamp(min=self._v_min, max=self._v_max)
            b = (t_z - self._v_min) / self._delta
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self._batch_size - 1) * self._atoms, self._batch_size
                )
                .long()
                .unsqueeze(1)
                .expand(self._batch_size, self._atoms)
                .to(self._device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self._device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        return proj_dist

    @torch.no_grad()
    def act(self, observation):
        # observation = torch.tensor(observation).to(self._device).float()

        # if not self._distributional:
        if self._training:
            if not self._learn_schedule.update():
                epsilon = 1.0
            else:
                epsilon = self._epsilon_schedule.update()
            if self._logger.update_step(self._timescale):
                self._logger.log_scalar("epsilon", epsilon, self._timescale)
        else:
            epsilon = 0

        if not self._epsilon_on:
            self._qnet.sample_noise()

        vectorized_observation = (
            torch.tensor(np.expand_dims(observation["vectorized"], axis=0))
            .to(self._device)
            .float()
        )
        qvals = self._qnet(vectorized_observation).cpu()

        if self._epsilon_on and self._rng.random() < epsilon:
            action = np.random.choice(observation["legal_moves_as_int"]).item()
        else:
            ilegal_moves = [
                x
                for x in list(np.arange(qvals.shape[1]))
                if x not in observation["legal_moves_as_int"]
            ]
            qvals[:, ilegal_moves] = qvals.min()
            action = torch.argmax(qvals).item()

        if self._logger.should_log(self._timescale) and self._state["episode_start"]:
            self._logger.log_scalar(
                "train_qval" if self._training else "test_qval",
                torch.max(qvals),
                self._timescale,
            )
            self._state["episode_start"] = False

        return action

    def update(self, update_info):
        """
        Updates the DQN agent.

        Args:
            update_info: dictionary containing all the necessary information to
            update the agent. Should contain a full transition, with keys for
            "observation", "action", "reward", "next_observation", and "done".
        """
        if update_info["done"]:
            self._state["episode_start"] = True

        # Add the most recent transition to the replay buffer.
        if self._training:
            self._replay_buffer.add(
                update_info["observation"]["vectorized"],
                update_info["action"],
                update_info["reward"],
                update_info["done"],
            )

        # Update the q network based on a sample batch from the replay buffer.
        # If the replay buffer doesn't have enough samples, catch the exception
        # and move on.
        if self._replay_buffer.size() > 0:
            batch = self._replay_buffer.sample(batch_size=self._batch_size)
            for key in batch:
                batch[key] = torch.tensor(batch[key]).to(self._device)

            # Compute predicted Q values
            self._optimizer.zero_grad()
            pred_qvals = self._qnet(batch["observation"])
            actions = batch["action"].long()

            if not self._distributional:
                pred_qvals = pred_qvals[torch.arange(pred_qvals.size(0)), actions]

                # Compute 1-step Q targets
                if self._double:
                    next_action = self._qnet(batch["next_observation"])
                else:
                    next_action = self._target_qnet(batch["next_observation"])

                _, next_action = torch.max(next_action, dim=1)
                next_qvals = self._target_qnet(batch["next_observation"])
                next_qvals = next_qvals[torch.arange(next_qvals.size(0)), next_action]

                q_targets = batch["reward"] + self._discount_rate * next_qvals * (
                    1 - batch["done"]
                )

                loss = self._loss_fn(pred_qvals, q_targets)

            else:
                if not self._epsilon_on:
                    self._qnet.sample_noise()
                    self._target_qnet.sample_noise()
                current_dist = self._qnet.dist(batch["observation"])
                log_p = torch.log(current_dist[range(self._batch_size), actions])
                target_prob = self.projection_distribution(batch)

                loss = -(target_prob * log_p).sum(1)
                loss = loss.mean()

            if self._logger.should_log(self._timescale):
                self._logger.log_scalar(
                    "train_loss" if self._training else "test_loss",
                    loss,
                    self._timescale,
                )
            if self._training:
                loss.backward()
                if self._grad_clip is not None:
                    torch.nn.utils.clip_grad_value_(
                        self._qnet.parameters(), self._grad_clip
                    )
                self._optimizer.step()

        # Update target network
        if self._training and self._target_net_update_schedule.update():
            self._update_target()
