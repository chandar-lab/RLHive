from typing import Tuple

import numpy as np
import torch
from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.utils import calculate_output_dim
from torch import nn


class NetPerActionDynaQModel(nn.Module):
    """
    Implements the model used in deep Dyna-Q algorithm.
    In this implementation there are multiple neural networks
        per each action for determining the next observations,
        actions, and terminated (termination signals).
    """

    def __init__(
        self,
        in_dim: Tuple[int],
        act_dim: int,
        observation_encoder_net: FunctionApproximator,
        reward_encoder_net: FunctionApproximator,
        terminated_encoder_net: FunctionApproximator,
        add_obs: bool = True,
        obs_linear_layer: bool = False,
    ):
        """
        Args:
            in_dim (tuple[int]): The shape of input observations.
            act_dim (int): The number of various possible actions.
            observation_encoder_net (FunctionApproximator): A network that outputs the
                representations that will be used to compute next observations.
            reward_encoder_net (FunctionApproximator): A network that outputs the
                representations that will be used to compute rewards.
            terminated_encoder_net (FunctionApproximator): A network that outputs
                the representations that will be used to compute
                episode terminations of episodes.
            add_obs (bool): Whether to sum up the observation predictions to the
                current observations or not.
            obs_linear_layer (bool): Whether to use a linear layer at the end of the
                observation predictor.
        """
        super().__init__()

        self._act_dim = act_dim
        self._add_obs = add_obs
        self._obs_linear_layer = obs_linear_layer

        # Observations
        self._obs_predictor = observation_encoder_net(in_dim)
        if self._obs_linear_layer:
            obs_predictor_out_dim = np.prod(
                calculate_output_dim(self._obs_predictor, in_dim)
            )
            self._obs_predictor = nn.ModuleList(
                [
                    nn.Sequential(
                        self._obs_predictor,
                        nn.Linear(obs_predictor_out_dim, np.prod(in_dim)),
                    )
                    for _ in range(self._act_dim)
                ]
            )

        # Rewards
        self._reward_predictor = reward_encoder_net(in_dim)
        reward_predictor_out_dim = np.prod(
            calculate_output_dim(self._reward_predictor, in_dim)
        )
        self._reward_predictor = nn.ModuleList(
            [
                nn.Sequential(
                    self._reward_predictor, nn.Linear(reward_predictor_out_dim, 1)
                )
                for _ in range(self._act_dim)
            ]
        )

        # Episode Terminations
        self._terminated_predictor = terminated_encoder_net(in_dim)
        terminated_predictor_out_dim = np.prod(
            calculate_output_dim(self._terminated_predictor, in_dim)
        )
        self._terminated_predictor = nn.ModuleList(
            [
                nn.Sequential(
                    self._terminated_predictor,
                    nn.Linear(terminated_predictor_out_dim, 1),
                    nn.Sigmoid(),
                )
                for _ in range(self._act_dim)
            ]
        )

    def forward(self, obs, actions):
        batch_size = obs.shape[0]

        # Observations
        obs_pred_list = []
        for a in range(self._act_dim):
            obs_pred_list.append(self._obs_predictor[a](obs))
        obs_pred = torch.stack(obs_pred_list, dim=len(obs_pred_list[0].shape))[
            range(batch_size), :, actions
        ]
        if self._add_obs:
            obs_pred = obs_pred + obs

        # Rewards
        reward_pred_list = []
        for a in range(self._act_dim):
            reward_pred_list.append(self._reward_predictor[a](obs))
        reward_pred = torch.stack(reward_pred_list, dim=len(reward_pred_list[0].shape))[
            range(batch_size), :, actions
        ]

        # Episode Terminations
        terminated_pred_list = []
        for a in range(self._act_dim):
            terminated_pred_list.append(self._terminated_predictor[a](obs))
        terminated_pred = torch.stack(
            terminated_pred_list, dim=len(terminated_pred_list[0].shape)
        )[range(batch_size), :, actions]

        return obs_pred.squeeze(), reward_pred.squeeze(), terminated_pred.squeeze()


class ActionInMiddleDynaQModel(nn.Module):
    """
    Implements the model used in nonlinear Dyna Q algorithm.
    In this implementation the action is added to the representations
    given by the encoder. Then, the final predictor would give the
    next observations, rewards, and episode terminations.
    """

    def __init__(
        self,
        in_dim: Tuple[int],
        act_dim: int,
        observation_encoder_net: FunctionApproximator,
        observation_predictor_net: FunctionApproximator,
        reward_encoder_net: FunctionApproximator,
        reward_predictor_net: FunctionApproximator,
        terminated_encoder_net: FunctionApproximator,
        terminated_predictor_net: FunctionApproximator,
        add_obs: bool = True,
        obs_linear_layer: bool = False,
    ):
        """
        Args:
            in_dim (tuple[int]): The shape of input observations.
            act_dim (int): The number of various possible actions.
            observation_encoder_net (FunctionApproximator): A network that outputs the
                representations that will be used to compute next observations.
            observation_predictor_net (FunctionApproximator): A network that takes in
                the representations and outputs
                the predictions for the next observations.
            reward_encoder_net (FunctionApproximator): A network that outputs the
                representations that will be used to compute rewards.
            reward_predictor_net (FunctionApproximator): A network that takes in
                the representations and outputs the predictions for the rewards.
            terminated_encoder_net (FunctionApproximator): A network that outputs the
                representations that will be used to compute episode terminations.
            terminated_predictor_net (FunctionApproximator): A network that takes in
                the representations and outputs the predictions
                for the episode terminations.
            add_obs (bool): Whether to sum up the observation predictions to the
                current observations or not.
            obs_linear_layer (bool): Whether to use a linear layer at the end of the
                observation predictor.
        """
        super().__init__()

        self._act_dim = act_dim
        self._add_obs = add_obs
        self._obs_linear_layer = obs_linear_layer

        # Observations
        self._obs_encoder = observation_encoder_net(in_dim)
        obs_predictor_in_dim = (
            np.prod(calculate_output_dim(self._obs_encoder, in_dim)) + 1
        )
        self._obs_predictor = observation_predictor_net(obs_predictor_in_dim)
        if self._obs_linear_layer:
            obs_predictor_out_dim = np.prod(
                calculate_output_dim(self._obs_predictor, (obs_predictor_in_dim,))
            )
            self._obs_predictor = nn.Sequential(
                self._obs_predictor, nn.Linear(obs_predictor_out_dim, np.prod(in_dim))
            )

        # Rewards
        self._reward_encoder = reward_encoder_net(in_dim)
        reward_predictor_in_dim = (
            np.prod(calculate_output_dim(self._reward_encoder, in_dim)) + 1
        )
        self._reward_predictor = reward_predictor_net(reward_predictor_in_dim)
        reward_predictor_out_dim = np.prod(
            calculate_output_dim(self._reward_predictor, (reward_predictor_in_dim,))
        )
        self._reward_predictor = nn.Sequential(
            self._reward_predictor, nn.Linear(reward_predictor_out_dim, 1)
        )

        # Episode Terminations
        self._terminated_encoder = terminated_encoder_net(in_dim)
        terminated_predictor_in_dim = (
            np.prod(calculate_output_dim(self._terminated_encoder, in_dim)) + 1
        )
        self._terminated_predictor = terminated_predictor_net(
            terminated_predictor_in_dim
        )
        terminated_predictor_out_dim = np.prod(
            calculate_output_dim(
                self._terminated_predictor, (terminated_predictor_in_dim,)
            )
        )
        self._terminated_predictor = nn.Sequential(
            self._terminated_predictor,
            nn.Linear(terminated_predictor_out_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, obs, actions):
        while len(actions.shape) < 2:
            actions = actions.unsqueeze(1)
        # Observations
        obs_pred = self._obs_encoder(obs)
        obs_pred = torch.cat((obs_pred.flatten(start_dim=1), actions), dim=1)
        obs_pred = self._obs_predictor(obs_pred)
        if self._add_obs:
            obs_pred = obs_pred + obs

        # Rewards
        reward_pred = self._reward_encoder(obs)
        reward_pred = torch.cat((reward_pred.flatten(start_dim=1), actions), dim=1)
        reward_pred = self._reward_predictor(reward_pred)

        # Episode Terminations
        terminated_pred = self._terminated_encoder(obs)
        terminated_pred = torch.cat(
            (terminated_pred.flatten(start_dim=1), actions), dim=1
        )
        terminated_pred = self._terminated_predictor(terminated_pred)

        return obs_pred.squeeze(), reward_pred.squeeze(), terminated_pred.squeeze()
