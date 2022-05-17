from typing import Tuple
import numpy as np
import torch
from torch import nn

from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.utils import calculate_output_dim


class ActionInMiddleDynaQModel(nn.Module):
    """Implements the model used in nonlinear Dyna Q algorithm.
    In this implementation the action is added to the representations
    given by the encoder. Then, the final predictor would give the
    next observations, rewards, and episode terminations (done).
    """

    def __init__(
        self,
        in_dim: Tuple[int],
        observation_encoder_net: FunctionApproximator,
        observation_predictor_net: FunctionApproximator,
        reward_encoder_net: FunctionApproximator,
        reward_predictor_net: FunctionApproximator,
        done_encoder_net: FunctionApproximator,
        done_predictor_net: FunctionApproximator,
    ):
        """
        Args:
            in_dim (tuple[int]): The shape of input observations.
            observation_encoder_net (FunctionApproximator): A network that outputs the
                representations that will be used to compute next observations.
            observation_predictor_net (FunctionApproximator): A network that takes in
                the representations and outputs the predictions for the next observations.
            reward_encoder_net (FunctionApproximator): A network that outputs the
                representations that will be used to compute rewards.
            reward_predictor_net (FunctionApproximator): A network that takes in
                the representations and outputs the predictions for the rewards.
            done_encoder_net (FunctionApproximator): A network that outputs the
                representations that will be used to compute episode terminations (done).
            done_predictor_net (FunctionApproximator): A network that takes in
                the representations and outputs the predictions for the episode terminations (done).
        """
        super().__init__()

        # TODO Fix the issue with the predictors, they should be output size agnostic

        # Observations
        self._obs_encoder = observation_encoder_net(in_dim)
        obs_predictor_in_dim = (
            np.prod(calculate_output_dim(self._obs_encoder, in_dim)) + 1
        )
        self._obs_predictor = observation_predictor_net(obs_predictor_in_dim)
        obs_predictor_out_dim = np.prod(
            calculate_output_dim(self._obs_predictor, (obs_predictor_in_dim, ))
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
            calculate_output_dim(self._reward_predictor, (reward_predictor_in_dim, ))
        )
        self._reward_predictor = nn.Sequential(
            self._reward_predictor, nn.Linear(reward_predictor_out_dim, 1)
        )

        # Episode Terminations
        self._done_encoder = done_encoder_net(in_dim)
        done_predictor_in_dim = (
            np.prod(calculate_output_dim(self._done_encoder, in_dim)) + 1
        )
        self._done_predictor = done_predictor_net(done_predictor_in_dim)
        done_predictor_out_dim = np.prod(
            calculate_output_dim(self._done_predictor, (done_predictor_in_dim, ))
        )
        self._done_predictor = nn.Sequential(
            self._done_predictor, nn.Linear(done_predictor_out_dim, 1), nn.Sigmoid()
        )

    def forward(self, obs, actions):
        while len(actions.shape) < len(obs.shape):
            actions = actions.unsqueeze(1)
        # Observations
        obs_pred = self._obs_encoder(obs)
        obs_pred = torch.cat((obs_pred.flatten(start_dim=1), actions), dim=1)
        obs_pred = self._obs_predictor(obs_pred)

        # Rewards
        reward_pred = self._reward_encoder(obs)
        reward_pred = torch.cat((reward_pred.flatten(start_dim=1), actions), dim=1)
        reward_pred = self._reward_predictor(reward_pred)

        # Episode Terminations
        done_pred = self._done_encoder(obs)
        done_pred = torch.cat((done_pred.flatten(start_dim=1), actions), dim=1)
        done_pred = self._done_predictor(done_pred)

        return obs_pred.squeeze(), reward_pred.squeeze(), done_pred.squeeze()
