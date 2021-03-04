import copy
import numpy as np
import torch

from hive import replays
from hive.utils import logging, schedule
from .agent import Agent


class DQNAgent(Agent):
    """An agent implementing the DQN algorithm. Uses an epsilon greedy
    exploration policy
    """

    def __init__(
        self,
        qnet,
        env_spec,
        optimizer,
        replay_buffer=None,
        discount_rate=0.99,
        target_net_soft_update=False,
        target_net_update_fraction=0.05,
        target_net_update_schedule=None,
        epsilon_schedule=None,
        learn_schedule=None,
        rng=None,
        batch_size=32,
        device="cpu",
        logger=None,
    ):
        """
        Args:
            qnet: A network that outputs the q-values of the different actions
                for an input observation.
            env_spec: The specification of the environment the agent will be
                running in.
            optimizer: A function that takes in a list of parameters to optimize
                and returns the optimizer.
            replay_buffer: The replay buffer that the agent will push observations
                to and sample from during learning.
            discount_rate (float): A number between 0 and 1 specifying how much
                future rewards are discounted by the agent.
            target_net_soft_update (bool): Whether the target net parameters are 
                replaced by the qnet parameters completely or using a weighted
                average of the target net parameters and the qnet parameters.
            target_net_update_fraction (float): The weight given to the target
                net parameters in a soft update.
            target_net_update_schedule: Schedule determining how frequently the
                target net is updated.
            epsilon_schedule: Schedule determining the value of epsilon through
                the course of training.
            learn_schedule: Schedule determining when the learning process actually
                starts.
            rng: numpy random number generator.
            batch_size (int): The size of the batch sampled from the replay buffer
                during learning.
            device: Device on which all computations should be run.
            logger: Logger used to log agent's metrics.
        """
        self._qnet = qnet
        self._env_spec = env_spec
        # Should this be a copy or should we implement a more standard func approximator copy
        self._target_qnet = copy.deepcopy(self._qnet).requires_grad_(False)
        self._optimizer = optimizer(self._qnet.parameters())
        self._rng = rng
        if self._rng is None:
            self._rng = np.random.default_rng(seed=42)
        self._replay_buffer = replay_buffer
        if replay_buffer is None:
            self._replay_buffer = replays.CircularReplayBuffer(self._rng)
        self._discount_rate = discount_rate
        self._grad_clip = grad_clip
        self._target_net_soft_update = target_net_soft_update
        self._target_net_update_fraction = target_net_update_fraction
        self._device = torch.device(device)
        self._loss_fn = torch.nn.SmoothL1Loss()
        self._batch_size = batch_size
        self._logger = logger
        if self._logger is None:
            self._logger = logging.NullLogger()

        self._target_net_update_schedule = target_net_update_schedule
        if self._target_net_update_schedule is None:
            self._target_net_update_schedule = schedule.PeriodicSchedule(
                False, True, 10000
            )
        self._epsilon_schedule = epsilon_schedule
        if self._epsilon_schedule is None:
            self._epsilon_schedule = schedule.LinearSchedule(1, 0.1, 100000)

        self._learn_schedule = learn_schedule
        if self._learn_schedule is None:
            self._learn_schedule = schedule.SwitchSchedule(False, True, 5000)

    def train(self):
        """Changes the agent to training mode."""
        super().train()
        self._qnet.train()
        self._target_qnet.train()

    def eval(self):
        """Changes the agent to evaluation mode."""
        super().eval()
        self._qnet.eval()
        self._target_qnet.eval()

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
        else:
            epsilon = 0
        self._logger.update_step()
        if self._logger.should_log():
            self._logger.log_scalar("epsilon", epsilon)

        # Sample action. With epsilon probability choose random action,
        # otherwise select the action with the highest q-value.
        if self._rng.random() < epsilon:
            action = self._rng.integers(self._env_spec.act_dim)
        else:
            observation = torch.tensor(observation).to(self._device).float()
            qvals = self._qnet(observation).cpu()
            action = torch.argmax(qvals).numpy()
        return action

    def update(self, update_info):
        """
        Updates the DQN agent.

        Args:
            update_info: dictionary containing all the necessary information to
            update the agent. Should contain a full transition, with keys for
            "observation", "action", "reward", "next_observation", and "done".
        """

        # Add the most recent transition to the replay buffer.
        self._replay_buffer.add(
            (
                update_info["observation"],
                update_info["action"],
                update_info["reward"],
                update_info["next_observation"],
                update_info["done"],
            )
        )

        # Update the q network based on a sample batch from the replay buffer.
        # If the replay buffer doesn't have enough samples, catch the exception
        # and move on.
        try:
            batch = self._replay_buffer.sample(batch_size=self._batch_size)
            for key in batch:
                batch[key] = torch.tensor(batch[key]).to(self._device)

            # Compute predicted Q values
            self._optimizer.zero_grad()
            pred_qvals = self._qnet(batch["observations"])
            actions = batch["actions"].long()
            pred_qvals = pred_qvals[torch.arange(pred_qvals.size(0)), actions]

            # Compute 1-step Q targets
            next_qvals = self._target_qnet(batch["next_observations"])
            next_qvals, _ = torch.max(next_qvals, dim=1)

            q_targets = batch["rewards"] + self._discount_rate * next_qvals * (
                1 - batch["done"]
            )

            loss = self._loss_fn(pred_qvals, q_targets)
            if self._logger.should_log():
                self._logger.log_scalar("loss", loss)

            loss.backward()
            self._optimizer.step()

        except IndexError:
            pass

        # Update target network
        if self._target_net_update_schedule.update():
            self._update_target()

    def _update_target(self):
        if self._target_net_soft_update:
            target_params = self._target_qnet.state_dict()
            current_params = self._qnet.state_dict()
            for key in list(target_params.keys()):
                target_params[key] = (
                    (1 - self._target_net_update_fraction) * target_params[key]
                    + self._target_net_update_fraction * current_params[key]
                )
            self._target_qnet.load_state_dict(target_params)
        else:
            self._target_qnet.load_state_dict(self._qnet.state_dict())

    def save(self, dname):
        pass

    def load(self, dname):
        pass

