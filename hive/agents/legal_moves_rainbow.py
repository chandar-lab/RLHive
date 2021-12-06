from typing import Tuple

import numpy as np
import torch

from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.utils import InitializationFn
from hive.agents.rainbow import RainbowDQNAgent
from hive.replays.replay_buffer import BaseReplayBuffer
from hive.utils.logging import Logger
from hive.utils.schedule import Schedule
from hive.utils.utils import LossFn, OptimizerFn


class LegalMovesRainbowAgent(RainbowDQNAgent):
    """A Rainbow agent which supports games with legal actions."""

    def __init__(
        self,
        representation_net: FunctionApproximator,
        obs_dim: Tuple,
        act_dim: int,
        v_min: str = 0,
        v_max: str = 200,
        atoms: str = 51,
        optimizer_fn: OptimizerFn = None,
        loss_fn: LossFn = None,
        init_fn: InitializationFn = None,
        id: str = 0,
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
        learn_schedule: Schedule = None,
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
            qnet: A network that outputs the q-values of the different actions
                for an input observation.
            obs_dim: The dimension of the observations.
            act_dim: The number of actions available to the agent.
            v_min: minimum possible value of the value function.
            v_max: maximum possible value of the value function.
            atoms: number of atoms in the distributional DQN context.
            optimizer_fn: A function that takes in a list of parameters to optimize
                and returns the optimizer.
            init_fn: initializes the weights of qnet using create_init_weights_fn.
            id: ID used to create the timescale in the logger for the agent.
            replay_buffer: The replay buffer that the agent will push observations
                to and sample from during learning.
            discount_rate (float): A number between 0 and 1 specifying how much
                future rewards are discounted by the agent.
            n_step (int): n used in n-step returns to compute TD(n) targets.
            grad_clip (float): Gradients will be clipped to between
                [-grad_clip, gradclip]
            reward_clip (float): Rewards will be clipped to between
                [-reward_clip, reward_clip]
            target_net_soft_update (bool): Whether the target net parameters are
                replaced by the qnet parameters completely or using a weighted
                average of the target net parameters and the qnet parameters.
            target_net_update_fraction (float): The weight given to the target
                net parameters in a soft update.
            target_net_update_schedule: Schedule determining how frequently the
                target net is updated.
            update_period_schedule: Schedule determining how frequently
                the agent's net is updated.
            epsilon_schedule: Schedule determining the value of epsilon through
                the course of training.
            test_epsilon (float): epsilon (probability of choosing a random action)
                to be used during testing phase.
            learn_schedule: Schedule determining when the learning process actually
                starts.
            seed: Seed for numpy random number generator.
            batch_size (int): The size of the batch sampled from the replay buffer
                during learning.
            device: Device on which all computations should be run.
            logger: Logger used to log agent's metrics.
            log_frequency (int): How often to log the agent's metrics.
            noisy (bool): whether or not to use the noisy feature (from noisy DQN).
            std_init (float): standard deviation for initialization of noises in the
                case of noisy networks.
            double: whether or not to use the double feature (from double DQN).
            dueling: whether or not to use the dueling feature (from dueling DQN).
            distributional: whether or not to use the distributional
                feature (from distributional DQN).
            use_eps_greedy: whether or not to use epsilon greedy.
                Usually in case of noisy networks use_eps_greedy=False.
        """
        super().__init__(
            representation_net,
            obs_dim,
            act_dim,
            v_min=v_min,
            v_max=v_max,
            atoms=atoms,
            optimizer_fn=optimizer_fn,
            loss_fn=loss_fn,
            init_fn=init_fn,
            id=id,
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
            learn_schedule=learn_schedule,
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

    def create_q_networks(self, representation_net):
        """Creates the qnet and target qnet."""
        super().create_q_networks(representation_net)
        self._qnet = LegalMovesHead(self._qnet)
        self._target_qnet = LegalMovesHead(self._target_qnet)

    def preprocess_update_info(self, update_info):
        """Clips the reward in update_info."""
        preprocessed_update_info = {
            "observation": update_info["observation"]["observation"],
            "action": update_info["action"],
            "reward": update_info["reward"],
            "done": update_info["done"],
            "action_mask": action_encoding(update_info["observation"]["action_mask"]),
        }
        if "agent_id" in update_info:
            preprocessed_update_info["agent_id"] = int(update_info["agent_id"])
        return preprocessed_update_info

    def preprocess_update_batch(self, batch):
        for key in batch:
            batch[key] = torch.tensor(batch[key], device=self._device)
        return (
            (batch["observation"], batch["action_mask"]),
            (batch["next_observation"], batch["next_action_mask"]),
        )

    @torch.no_grad()
    def act(self, observation):
        if self._training:
            if not self._learn_schedule.get_value():
                epsilon = 1.0
            elif not self._use_eps_greedy:
                epsilon = 0.0
            else:
                epsilon = self._epsilon_schedule.update()
            if self._logger.update_step(self._timescale):
                self._logger.log_scalar("epsilon", epsilon, self._timescale)
        else:
            epsilon = self._test_epsilon

        vectorized_observation = torch.tensor(
            np.expand_dims(observation["observation"], axis=0), device=self._device
        ).float()
        legal_moves_as_int = [
            i for i, x in enumerate(observation["action_mask"]) if x == 1
        ]
        encoded_legal_moves = torch.tensor(
            action_encoding(observation["action_mask"]), device=self._device
        ).float()
        qvals = self._qnet(vectorized_observation, encoded_legal_moves).cpu()

        if self._rng.random() < epsilon:
            action = np.random.choice(legal_moves_as_int).item()
        else:
            action = torch.argmax(qvals).item()

        if (
            self._training
            and self._logger.should_log(self._timescale)
            and self._state["episode_start"]
        ):
            self._logger.log_scalar("train_qval", torch.max(qvals), self._timescale)
            self._state["episode_start"] = False

        return action


def action_encoding(action_mask):
    encoded_action_mask = np.zeros(action_mask.shape)
    encoded_action_mask[action_mask == 0] = -np.inf
    return encoded_action_mask


class LegalMovesHead(torch.nn.Module):
    def __init__(self, base_network):
        super().__init__()
        self.base_network = base_network

    def forward(self, x, legal_moves):
        x = self.base_network(x)
        return x + legal_moves

    def dist(self, x, legal_moves):
        return self.base_network.dist(x)
