import numpy as np
import torch

from hive.agents.dqn import DQNAgent


class HanabiDQNAgent(DQNAgent):
    """An agent implementing the DQN algorithm. Uses an epsilon greedy
    exploration policy
    """

    def __init__(
        self,
        qnet,
        obs_dim,
        act_dim,
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
    ):
        super().__init__(
            qnet,
            obs_dim,
            act_dim,
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
        )

    @torch.no_grad()
    def act(self, observation):
        """Returns the action for the agent. If in training mode, follows an epsilon
        greedy policy. Otherwise, returns the action with the highest q value."""

        # Determine and log the value of epsilon
        if self._training:
            if not self._learn_schedule.update():
                epsilon = 1.0
            else:
                epsilon = self._epsilon_schedule.update()
            if self._logger.update_step(self._timescale):
                self._logger.log_scalar("epsilon", epsilon, self._timescale)
        else:
            epsilon = 0

        # Sample action. With epsilon probability choose random action,
        # otherwise select the action with the highest q-value.
        vectorized_observation = (
            torch.tensor(np.expand_dims(observation["vectorized"], axis=0))
            .to(self._device)
            .float()
        )
        qvals = self._qnet(vectorized_observation).cpu()
        if self._rng.random() < epsilon:
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
            "observation", "action", "reward", and "done".
        """
        if update_info["done"]:
            self._state["episode_start"] = True

        # Add the most recent transition to the replay buffer.
        if self._training:
            self._replay_buffer.add(
                observation=update_info["observation"]["vectorized"],
                action=update_info["action"],
                reward=update_info["reward"],
                done=update_info["done"],
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
            pred_qvals = pred_qvals[torch.arange(pred_qvals.size(0)), actions]

            # Compute 1-step Q targets
            next_qvals = self._target_qnet(batch["next_observation"])
            next_qvals, _ = torch.max(next_qvals, dim=1)

            q_targets = batch["reward"] + self._discount_rate * next_qvals * (
                1 - batch["done"]
            )

            loss = self._loss_fn(pred_qvals, q_targets)
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
