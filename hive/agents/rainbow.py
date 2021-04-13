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
from hive.agents.qnets import get_qnet


class RainbowDQNAgent(Agent):
    """An agent implementing the DQN algorithm. Uses an epsilon greedy
    exploration policy
    """

    def __init__(
        self,
        qnet,
        # obs_dim,
        # act_dim,
        env_spec,
        v_min,
        v_max,
        atoms,
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
        dueling=True,
        distributional=True,
        noisy=True,
    ):
        """
        Args:
            qnet: A network that outputs the q-values of the different actions
                for an input observation.
            obs_dim: The dimension of the observations.
            act_dim: The number of actions available to the agent.
            optimizer_fn: A function that takes in a list of parameters to optimize
                and returns the optimizer.
            id: ID used to create the timescale in the logger for the agent.
            replay_buffer: The replay buffer that the agent will push observations
                to and sample from during learning.
            discount_rate (float): A number between 0 and 1 specifying how much
                future rewards are discounted by the agent.
            grad_clip (float): Gradients will be clipped to between 
                [-grad_clip, gradclip]
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
            seed: Seed for numpy random number generator.
            batch_size (int): The size of the batch sampled from the replay buffer
                during learning.
            device: Device on which all computations should be run.
            logger: Logger used to log agent's metrics.
            log_frequency (int): How often to log the agent's metrics.
        """
        # super().__init__(obs_dim=obs_dim, act_dim=act_dim, id=f"dqn_agent_{id}")
        if isinstance(qnet, dict):
            qnet["kwargs"]["in_dim"] = env_spec.obs_dim
            qnet["kwargs"]["out_dim"] = env_spec.act_dim

        qnet["kwargs"]["noisy"] = noisy
        qnet["kwargs"]["dueling"] = dueling

        if distributional:
            qnet["name"] = 'DistributionalMLP'
        else:
            qnet["name"] = 'SimpleMLP'

        self._qnet = get_qnet(qnet)
        self._target_qnet = copy.deepcopy(self._qnet).requires_grad_(False)

        self._noisy = noisy
        self._double = double
        self._dueling = dueling
        self._distributional = distributional

        optimizer_fn = get_optimizer_fn(optimizer_fn)
        if optimizer_fn is None:
            optimizer_fn = torch.optim.Adam
        self._optimizer = optimizer_fn(self._qnet.parameters())
        self._rng = np.random.default_rng(seed=seed)
        self._replay_buffer = get_replay(replay_buffer)
        if self._replay_buffer is None:
            self._replay_buffer = CircularReplayBuffer(np.random.default_rng(seed=seed))
        self._discount_rate = discount_rate
        self._grad_clip = grad_clip
        self._target_net_soft_update = target_net_soft_update
        self._target_net_update_fraction = target_net_update_fraction
        self._device = torch.device(device)
        self._loss_fn = torch.nn.SmoothL1Loss()
        self._batch_size = batch_size
        self._logger = get_logger(logger)
        if self._logger is None:
            self._logger = NullLogger()
        self._timescale = self.id
        self._logger.register_timescale(
            self._timescale, PeriodicSchedule(False, True, log_frequency)
        )
        self._target_net_update_schedule = get_schedule(target_net_update_schedule)
        if self._target_net_update_schedule is None:
            self._target_net_update_schedule = PeriodicSchedule(False, True, 10000)
        self._epsilon_schedule = get_schedule(epsilon_schedule)
        if self._epsilon_schedule is None:
            self._epsilon_schedule = LinearSchedule(1, 0.1, 100000)

        self._learn_schedule = get_schedule(learn_schedule)
        if self._learn_schedule is None:
            self._learn_schedule = SwitchSchedule(False, True, 5000)

        self._state = {"episode_start": True}
        self._training = False

        self._dueling = dueling
        self._double = double

        self._atoms = atoms
        self._v_min = v_min
        self._v_max = v_max
        self._supports = torch.linspace(self._v_min, self._v_max, self._atoms).view(1, 1, self._atoms).to(self._device)
        self._delta = (self._v_max - self._v_min) / (self._atoms - 1)
        self._nsteps = 1

    def get_max_next_state_action(self, next_states):
        next_dist = self._qnet(next_states) * self._supports
        return next_dist.sum(dim=2).max(1)[1].view(next_states.size(0), 1, 1).expand(-1, -1, self._atoms)

    def projection_distribution(self, batch):
        batch_obs = batch["observations"]
        batch_action = batch["actions"].long()
        batch_next_obs = batch["next_observations"]
        batch_reward = batch["rewards"]
        batch_not_done = 1 - batch["done"]

        with torch.no_grad():
            max_next_dist = torch.zeros((self._batch_size, 1, self._atoms), device=self._device,
                                        dtype=torch.float) + 1./self._atoms

            # TO-DO: if not empty_next_state_values:
            max_next_action = self.get_max_next_state_action(batch_next_obs)
            self._target_qnet.sample_noise()
            max_next_dist[batch_not_done] = self._target_qnet(batch_next_obs).gather(1, max_next_action)
            max_next_dist = max_next_dist.squeeze()

            Tz = batch_reward.view(-1, 1) + (self._discount_rate ** self._nsteps) * self._supports.view(1, -1) * batch_not_done.to(
                torch.float).view(-1, 1)
            Tz = Tz.clamp(self._v_min, self._v_max)
            b = (Tz - self._v_min) / self._delta
            l = b.floor().to(torch.int64)
            u = b.ceil().to(torch.int64)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self._atoms - 1)) * (l == u)] += 1

            offset = torch.linspace(0, (self._batch_size - 1) * self._atoms, self._batch_size).unsqueeze(dim=1).expand(
                self._batch_size, self._atoms).to(batch_action)
            m = batch_obs.new_zeros(self._batch_size, self._atoms)
            m.view(-1).index_add_(0, (l + offset).view(-1),
                                  (max_next_dist * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1),
                                  (max_next_dist * (b - l.float())).view(-1))

        return m



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
        observation = torch.tensor(observation).to(self._device).float()
        self._qnet.sample_noise()
        a = self._qnet(observation) * self._supports
        a = a.sum(dim=2).max(1)[1].view(1,1)
        action = a.item()
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

            if not self._distributional:
                pred_qvals = pred_qvals[torch.arange(pred_qvals.size(0)), actions]

                # Compute 1-step Q targets
                if self._double:
                    next_action = self._qnet(batch["next_observations"])
                else:
                    next_action = self._target_qnet(batch["next_observations"])

                _, next_action = torch.max(next_action, dim=1)
                next_qvals = self._target_qnet(batch["next_observations"])
                next_qvals = next_qvals[torch.arange(next_qvals.size(0)), next_action]

                q_targets = batch["rewards"] + self._discount_rate * next_qvals * (
                    1 - batch["done"]
                )

                loss = self._loss_fn(pred_qvals, q_targets)

            else:
                print("distributional agent WIP")



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

        except IndexError:
            pass

        # Update target network
        if self._training and self._target_net_update_schedule.update():
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
        torch.save(
            {
                "qnet": self._qnet.state_dict(),
                "target_qnet": self._target_qnet.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "learn_schedule": self._learn_schedule,
                "epsilon_schedule": self._epsilon_schedule,
                "target_net_update_schedule": self._target_net_update_schedule,
                "rng": self._rng,
            },
            os.path.join(dname, "agent.pt"),
        )
        replay_dir = os.path.join(dname, "replay")
        create_folder(replay_dir)
        self._replay_buffer.save(replay_dir)

    def load(self, dname):
        checkpoint = torch.load(os.path.join(dname, "agent.pt"))
        self._qnet.load_state_dict(checkpoint["qnet"])
        self._target_qnet.load_state_dict(checkpoint["target_qnet"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])
        self._learn_schedule = checkpoint["learn_schedule"]
        self._epsilon_schedule = checkpoint["epsilon_schedule"]
        self._target_net_update_schedule = checkpoint["target_net_update_schedule"]
        self._rng = checkpoint["rng"]
        self._replay_buffer.load(os.path.join(dname, "replay"))
