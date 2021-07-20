from hive.agents.qnets.base import FunctionApproximator
import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Callable

from hive.utils.utils import create_folder, get_optimizer_fn
from hive.utils.logging import Logger, NullLogger, get_logger

from hive.utils.schedule import (
    PeriodicSchedule,
    LinearSchedule,
    SwitchSchedule,
    get_schedule,
    Schedule,
)
from hive.agents.agent import Agent
from hive.agents.qnets import get_qnet
from hive.utils.utils import OptimizerFn


import time

class REINFORCEAgent(Agent):
    """An agent implementing the REINFORCE algorithm. Uses an epsilon greedy
    exploration policy
    """

    def __init__(
        self,
        qnet: FunctionApproximator,
        obs_dim: Tuple,
        act_dim: int,
        optimizer_fn: OptimizerFn = None,
        id: str = 0,
        discount_rate: float = 0.99,
        grad_clip: float = None,
        epsilon_schedule: Schedule = None,
        seed: int = 42,
        device: str = "cpu",
        stack_size=1,
        logger: Logger = None,
    ):
        """
        Args:
            qnet: A network that outputs the q-values of the different actions
                for an input observation.
            obs_dim: The dimension of the observations.
            act_dim: The number of actions available to the agent.
            optimizer_fn: A function that takes in a list of parameters to optimize
                and returns the optimizer.
            id: identifier of the agent
            discount_rate (float): A number between 0 and 1 specifying how much
                future rewards are discounted by the agent.
            grad_clip (float): Gradients will be clipped to between
                [-grad_clip, gradclip]
            epsilon_schedule: Schedule determining the value of epsilon through
                the course of training.
            seed: Seed for numpy random number generator.
            device: Device on which all computations should be run.
            stack_size: number of previous observations to stack to give as input
                which is needed because no replay buffer is used here
            logger removed for now (set to -1)
        """
        super().__init__(obs_dim=obs_dim, act_dim=act_dim, id=id)
        self._qnet = qnet(self._obs_dim, self._act_dim).to(device)
        if optimizer_fn is None:
            optimizer_fn = torch.optim.Adam
        self._optimizer = optimizer_fn(self._qnet.parameters())
        self._rng = np.random.default_rng(seed=seed)
        self._discount_rate = discount_rate
        self._grad_clip = grad_clip
        self._device = torch.device(device)
        # self._loss_fn = torch.nn.SmoothL1Loss()
        self._epsilon_schedule = epsilon_schedule
        if self._epsilon_schedule is None:
            self._epsilon_schedule = LinearSchedule(1, 0.1, 100000)

        self._training = False

        self._stack_size = stack_size
        self._episode_info = {"observation": torch.tensor([]), "action": [], "reward": []}

        #adding small constant to prevent numerical errors in normalization of return
        self.eps = np.finfo(np.float32).eps.item()

    def train(self):
        """Changes the agent to training mode."""
        super().train()
        self._qnet.train()

    def eval(self):
        """Changes the agent to evaluation mode."""
        super().eval()
        self._qnet.eval()

    @torch.no_grad()
    def act(self, observation):
        """Returns the action for the agent. If in training mode, follows an epsilon
        greedy policy. Otherwise, returns the action with the highest q value."""

        # Determine and log the value of epsilon
        if self._training:
            epsilon = self._epsilon_schedule.update()
        else:
            epsilon = 0.0

        # Sample action. With epsilon probability choose random action,
        # otherwise select the action with the highest q-value.
        observation = (
            torch.tensor(np.expand_dims(observation, axis=0)).to(self._device).float()
        )
        qvals = self._qnet(observation).cpu()
        if self._rng.random() < epsilon:
            action = self._rng.integers(self._act_dim)
        else:
            action = torch.argmax(qvals).numpy()

        return action

    def update(self, update_info):
        """
        Updates the agent with the REINFORCE update.

        Args:
            update_info: dictionary containing all the necessary information to
            update the agent. Should contain a full transition, with keys for
            "observation", "action", "reward", and "done".
        """
        observation = update_info["observation"]
        action = update_info["action"].item()
        reward = update_info["reward"]

        if update_info["done"]:
            #Add (last) update_info to _episode_info
            #Perform reinforce update with _episode_info
            #Reset _episode_info info to empty
            self._episode_info["observation"] = torch.cat((self._episode_info["observation"], torch.tensor(np.expand_dims(observation, axis=0))), dim=0)
            self._episode_info["action"].append(action)
            self._episode_info["reward"].append(reward)

            if self._training:
                #construct stacked observation input
                obs_shape = self._episode_info["observation"].shape
                stacked_observations = torch.tensor([])
                
                for i in range(obs_shape[0]):
                    if i < self._stack_size-1:
                        zero_padding = torch.zeros([obs_shape[1]*(self._stack_size-i-1)] + list(obs_shape[2:]))
                        stacked_observation = torch.cat([zero_padding] + [self._episode_info["observation"][j] for j in range(0, i+1)], dim=0)
                    else:
                        stacked_observation = torch.cat([self._episode_info["observation"][j] for j in range(i+1-self._stack_size, i+1)], dim=0)
                    stacked_observations = torch.cat((stacked_observations, stacked_observation.unsqueeze(0)), dim=0)
                stacked_observations = stacked_observations.to(self._device)

                #perform reinforce update
                # Compute predicted Q values
                self._optimizer.zero_grad()
                pred_qvals = self._qnet(stacked_observations)

                actions = torch.tensor(self._episode_info["action"]).long().to(self._device)

                probs = F.softmax(pred_qvals, dim=1)
                prob_dist = Categorical(probs)
                log_probs = prob_dist.log_prob(actions)

                Ret = 0.0
                returns = []
                for rew in self._episode_info["reward"][::-1]:
                    Ret = rew + self._discount_rate*Ret
                    returns.insert(0, Ret)
                returns = torch.tensor(returns).float().to(self._device)
                returns = (returns - returns.mean()) / (returns.std() + self.eps)

                policy_loss = -torch.dot(log_probs, returns)

                policy_loss.backward()
                if self._grad_clip is not None:
                    torch.nn.utils.clip_grad_value_(
                        self._qnet.parameters(), self._grad_clip
                    )
                self._optimizer.step()

            #reset episode buffer
            self._episode_info = {"observation": torch.tensor([]), "action": [], "reward": []}

        else:
            #Add (non-last) update_info to episode info
            self._episode_info["observation"] = torch.cat((self._episode_info["observation"], torch.tensor(np.expand_dims(observation, axis=0))), dim=0)
            self._episode_info["action"].append(action)
            self._episode_info["reward"].append(reward)

    def save(self, dname):
        torch.save(
            {
                "qnet": self._qnet.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "epsilon_schedule": self._epsilon_schedule,
                "rng": self._rng,
            },
            os.path.join(dname, "agent.pt"),
        )

    def load(self, dname):
        checkpoint = torch.load(os.path.join(dname, "agent.pt"))
        self._qnet.load_state_dict(checkpoint["qnet"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])
        self._epsilon_schedule = checkpoint["epsilon_schedule"]
        self._rng = checkpoint["rng"]