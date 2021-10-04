import copy
from typing import Tuple, Callable
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from hive.agents.rainbow import RainbowDQNAgent
from hive.agents.qnets.base import FunctionApproximator
from hive.replays.replay_buffer import BaseReplayBuffer
from hive.utils.logging import Logger
from hive.utils.schedule import Schedule
from hive.utils.utils import OptimizerFn


class HanabiRainbowAgent(RainbowDQNAgent):
    """A Hanabi agent implementing the Rainbow algorithm."""

    def __init__(
        self,
        qnet: FunctionApproximator,
        obs_dim: Tuple,
        act_dim: int,
        v_min: str = 0,
        v_max: str = 200,
        atoms: str = 51,
        optimizer_fn: OptimizerFn = None,
        id: str = 0,
        replay_buffer: BaseReplayBuffer = None,
        discount_rate: float = 0.99,
        n_step: int = 1,
        grad_clip: float = None,
        reward_clip: float = None,
        target_net_soft_update: bool = False,
        target_net_update_fraction: float = 0.05,
        target_net_update_schedule: Schedule = None,
        update_period_schedule: Schedule = None,
        epsilon_schedule: Schedule = None,
        learn_schedule: Schedule = None,
        seed: int = 42,
        batch_size: int = 32,
        device: str = "cpu",
        logger: Logger = None,
        log_frequency: int = 100,
        noisy=True,
        std_init=0.5,
        double: bool = True,
        dueling: bool = True,
        distributional: bool = True,
        use_eps_greedy: bool = False,
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
            n_step=n_step,
            grad_clip=grad_clip,
            reward_clip=reward_clip,
            target_net_soft_update=target_net_soft_update,
            target_net_update_fraction=target_net_update_fraction,
            target_net_update_schedule=target_net_update_schedule,
            update_period_schedule=update_period_schedule,
            epsilon_schedule=epsilon_schedule,
            learn_schedule=learn_schedule,
            seed=seed,
            batch_size=batch_size,
            device=device,
            logger=logger,
            log_frequency=log_frequency,
            noisy=noisy,
            std_init=std_init,
            double=double,
            dueling=dueling,
            distributional=distributional,
            use_eps_greedy=use_eps_greedy,
        )

    def create_q_networks(self, qnet, device):
        super(HanabiRainbowAgent, self).create_q_networks(qnet, device)

        self._qnet = HanabiHead(self._qnet).to(device=device)
        self._target_qnet = HanabiHead(self._target_qnet).to(device=device)


class HanabiHead(nn.Module):
    def __init__(self, qnet):
        super().__init__()
        self._qnet = qnet

    def forward(self, x, action_mask):
        x = self._qnet(x)
        return x + action_mask

    def dist(self, x):
        return self._qnet.dist(x)
