import abc

import numpy as np
import torch
from torch import nn

from hive.agents.qnets.mlp import MLPNetwork
from hive.agents.qnets.utils import calculate_output_dim
from hive.utils.registry import Registrable, registry


class SequenceFn(Registrable, nn.Module):
    """A wrapper for callables that produce sequence functions."""

    @classmethod
    def type_name(cls):
        return "SequenceFn"

    @abc.abstractmethod
    def init_hidden(self, batch_size):
        raise NotImplementedError

    def get_hidden_size(self):
        return None


class LSTMModel(SequenceFn):
    """
    A multi-layer long short-term memory (LSTM) RNN.
    """

    def __init__(
        self,
        rnn_input_size=256,
        rnn_hidden_size=128,
        num_rnn_layers=1,
        batch_first=True,
    ):
        """
        Args:
            rnn_input_size (int): The number of expected features in the input x.
            rnn_hidden_size (int):  The number of features in the hidden state h.
            num_rnn_layers (int): Number of recurrent layers.
            batch_first (bool): If True, then the input and output tensors are
            provided as (batch, seq, feature) instead of (seq, batch, feature).
        """
        super().__init__()
        self._rnn_hidden_size = rnn_hidden_size
        self._num_rnn_layers = num_rnn_layers
        self.core = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=self._rnn_hidden_size,
            num_layers=self._num_rnn_layers,
            batch_first=batch_first,
        )
        self._device = next(self.core.parameters()).device

    def forward(self, x, hidden_state=None):
        x, hidden_state = self.core(x, (hidden_state[0], hidden_state[1]))
        return x, hidden_state

    def to(self, device, *args, **kwargs):
        self._device = device
        return super().to(device, *args, **kwargs)

    def forward(self, x, hidden_state=None):
        x, hidden_state = self.core(x, hidden_state)
        return x, hidden_state

    def _update_device(self):
        self._device = next(self.core.parameters()).device

    def init_hidden(self, batch_size):
        return torch.zeros(
            (2, self._num_rnn_layers, batch_size, self._rnn_hidden_size),
            dtype=torch.float32,
            device=self._device,
        )

    def get_hidden_size(self):
        return (2, self._num_rnn_layers, self._rnn_hidden_size)

    def update_device(self):
        self._update_device()


class GRUModel(SequenceFn):
    """
    A multi-layer gated recurrent unit (GRU) RNN.
    """

    def __init__(
        self,
        rnn_input_size=256,
        rnn_hidden_size=128,
        num_rnn_layers=1,
        batch_first=True,
    ):
        """
        Args:
            rnn_input_size (int): The number of expected features in the input x.
            rnn_hidden_size (int):  The number of features in the hidden state h.
            num_rnn_layers (int): Number of recurrent layers.
            batch_first (bool): If True, then the input and output tensors are
            provided as (batch, seq, feature) instead of (seq, batch, feature).
        """
        super().__init__()
        self._rnn_hidden_size = rnn_hidden_size
        self._num_rnn_layers = num_rnn_layers
        self.core = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=self._rnn_hidden_size,
            num_layers=self._num_rnn_layers,
            batch_first=batch_first,
        )
        self._device = next(self.core.parameters()).device

    def forward(self, x, hidden_state=None):
        x, hidden_state = self.core(x, hidden_state)
        return x, hidden_state

    def to(self, device, *args, **kwargs):
        self._device = device
        return super().to(device, *args, **kwargs)

    def forward(self, x, hidden_state=None):
        x, hidden_state = self.core(x, hidden_state)
        return x, hidden_state

    def _update_device(self):
        self._device = next(self.core.parameters()).device

    def init_hidden(self, batch_size):
        return torch.zeros(
            (self._num_rnn_layers, batch_size, self._rnn_hidden_size),
            dtype=torch.float32,
            device=self._device,
        )

    def get_hidden_size(self):
        return (self._num_rnn_layers, self._rnn_hidden_size)


class SequenceModel(Registrable, nn.Module):
    """
    Basic convolutional recurrent neural network architecture. Applies a number of
    convolutional layers (each followed by a ReLU activation), recurrent layers, and then
    feeds the output into an :py:class:`hive.agents.qnets.mlp.MLPNetwork`.

    Note, if :obj:`channels` is :const:`None`, the network created for the
    convolution portion of the architecture is simply an
    :py:class:`torch.nn.Identity` module. If :obj:`mlp_layers` is
    :const:`None`, the mlp portion of the architecture is an
    :py:class:`torch.nn.Identity` module.
    """

    def __init__(
        self,
        in_dim,
        representation_network: torch.nn.Module,
        sequence_fn: SequenceFn,
        mlp_layers=None,
        normalization_factor=255,
        noisy=False,
        std_init=0.5,
    ):
        """
        Args:
            in_dim (tuple): The tuple of observations dimension (channels, width,
                height).
            sequence_fn (SequenceFn): A sequence neural network that learns
                recurrent representation. Usually placed between the convolutional
                layers and mlp layers.
            normalization_factor (float | int): What the input is divided by before
                the forward pass of the network.
            noisy (bool): Whether the MLP part of the network will use
                :py:class:`~hive.agents.qnets.noisy_linear.NoisyLinear` layers or
                :py:class:`torch.nn.Linear` layers.
            std_init (float): The range for the initialization of the standard
                deviation of the weights in
                :py:class:`~hive.agents.qnets.noisy_linear.NoisyLinear`.
        """
        super().__init__()

        self._normalization_factor = normalization_factor
        self.representation_network = representation_network

        # RNN Layers
        conv_output_size = calculate_output_dim(self.representation_network, in_dim)
        self.sequence_fn = sequence_fn(
            rnn_input_size=np.prod(conv_output_size),
        )

        if mlp_layers is not None:
            # MLP Layers
            sequence_output_size, _ = calculate_output_dim(
                self.sequence_fn, conv_output_size
            )
            self.mlp = MLPNetwork(
                sequence_output_size,
                mlp_layers,
                noisy=noisy,
                std_init=std_init,
            )
        else:
            self.mlp = nn.Identity()

    def forward(self, x, hidden_state=None):
        B, L = x.shape[0], x.shape[1]
        x = x.reshape(B * L, *x.shape[2:])
        x = self.representation_network(x)
        x = x.view(B, L, -1)

        # Sequence models with hidden state
        if hidden_state is None:
            hidden_state = self.sequence_fn.init_hidden(batch_size=B)

        x, hidden_state = self.sequence_fn(x, hidden_state)
        out = self.mlp(x.reshape((B * L, -1)))
        return out, hidden_state

    def get_hidden_size(self):
        return self.sequence_fn.get_hidden_size()

    @classmethod
    def type_name(cls):
        return "SequenceModel"


class DRQNNetwork(nn.Module):
    """Implements the standard DRQN value computation. This module returns two outputs,
    which correspond to the two outputs from :obj:`base_network`. In particular, it
    transforms the first output from :obj:`base_network` with output dimension
    :obj:`hidden_dim` to dimension :obj:`out_dim`, which should be equal to the
    number of actions. The second output of this module is the second output from
    :obj:`base_network`, which is the hidden state that will be used as the initial
    hidden state when computing the next action in the trajectory.
    """

    def __init__(
        self,
        base_network: SequenceModel,
        hidden_dim: int,
        out_dim: int,
        linear_fn: nn.Module = None,
    ):
        """
        Args:
            base_network (torch.nn.Module): Backbone network that returns two outputs,
                one is the representation used to compute action values, and the
                other one is the hidden state used as input hidden state later.
            hidden_dim (int): Dimension of the output of the :obj:`network`.
            out_dim (int): Output dimension of the DRQN. Should be equal to the
                number of actions that you are computing values for.
            linear_fn (torch.nn.Module): Function that will create the
                :py:class:`torch.nn.Module` that will take the output of
                :obj:`network` and produce the final action values. If
                :obj:`None`, a :py:class:`torch.nn.Linear` layer will be used.
        """
        super().__init__()
        self.base_network = base_network
        self._linear_fn = linear_fn if linear_fn is not None else nn.Linear
        self.output_layer = self._linear_fn(hidden_dim, out_dim)

    def forward(self, x, hidden_state=None):
        x, hidden_state = self.base_network(x, hidden_state)

        x = x.flatten(start_dim=1)
        return self.output_layer(x), hidden_state

    def get_hidden_size(self):
        return self.base_network.get_hidden_size()

    def update_device(self):
        self._update_device()


class SequenceNetwork(nn.Module):
    """
    Basic convolutional recurrent neural network architecture. Applies a number of
    convolutional layers (each followed by a ReLU activation), recurrent layers, and then
    feeds the output into an :py:class:`hive.agents.qnets.mlp.MLPNetwork`.

    Note, if :obj:`channels` is :const:`None`, the network created for the
    convolution portion of the architecture is simply an
    :py:class:`torch.nn.Identity` module. If :obj:`mlp_layers` is
    :const:`None`, the mlp portion of the architecture is an
    :py:class:`torch.nn.Identity` module.
    """

    def __init__(
        self,
        in_dim,
        representation_network: torch.nn.Module,
        sequence_fn: SequenceFn,
        mlp_layers=None,
        normalization_factor=255,
        noisy=False,
        std_init=0.5,
    ):
        """
        Args:
            in_dim (tuple): The tuple of observations dimension (channels, width,
                height).
            sequence_fn (SequenceFn): A sequence neural network that learns
                recurrent representation. Usually placed between the convolutional
                layers and mlp layers.
            normalization_factor (float | int): What the input is divided by before
                the forward pass of the network.
            noisy (bool): Whether the MLP part of the network will use
                :py:class:`~hive.agents.qnets.noisy_linear.NoisyLinear` layers or
                :py:class:`torch.nn.Linear` layers.
            std_init (float): The range for the initialization of the standard
                deviation of the weights in
                :py:class:`~hive.agents.qnets.noisy_linear.NoisyLinear`.
        """
        super().__init__()

        self._normalization_factor = normalization_factor
        self.representation_network = representation_network

        # RNN Layers
        conv_output_size = calculate_output_dim(self.representation_network, in_dim)
        self.sequence_fn = sequence_fn(
            rnn_input_size=np.prod(conv_output_size),
        )

        if isinstance(self.sequence_fn.core, torch.nn.LSTM):
            self._rnn_type = "lstm"
        elif isinstance(self.sequence_fn.core, torch.nn.GRU):
            self._rnn_type = "gru"
        else:
            self._rnn_type = "none"

        if mlp_layers is not None:
            # MLP Layers
            self.mlp = MLPNetwork(
                sequence_fn.keywords["rnn_hidden_size"],
                mlp_layers,
                noisy=noisy,
                std_init=std_init,
            )
        else:
            self.mlp = nn.Identity()

    def forward(self, x, agent_traj_state=None):
        B, L = x.shape[0], x.shape[1]
        x = x.reshape(B * L, *x.shape[2:])
        x = self.representation_network(x)
        x = x.view(B, L, -1)

        # Sequence models with hidden state
        if self.sequence_fn._rnn_hidden_size is not None:
            if agent_traj_state is not None:
                assert "hidden_state" in agent_traj_state.keys()
                if isinstance(agent_traj_state["hidden_state"], dict):
                    hidden_state = tuple(
                        torch.tensor(array)
                        for array in agent_traj_state["hidden_state"].values()
                    )
                else:
                    hidden_state = tuple(
                        torch.tensor(array).view(
                            self.sequence_fn._num_rnn_layers, B, -1
                        )
                        for array in agent_traj_state.values()
                    )

            else:
                hidden_state = self.sequence_fn.init_hidden(batch_size=1)
                agent_traj_state = {}

            x, hidden_state = self.sequence_fn(x, hidden_state)
            agent_traj_state["hidden_state"] = self._unpack_hidden_state(hidden_state)

        else:
            x = self.sequence_fn(x)

        out = self.mlp(x.reshape((B * L, -1)))

        return out, agent_traj_state

    def _unpack_hidden_state(self, hidden_state):
        if self._rnn_type == "lstm":
            hidden_state = {
                "hidden_state": hidden_state[0].detach().cpu().numpy(),
                "cell_state": hidden_state[1].detach().cpu().numpy(),
            }

        elif self._rnn_type == "gru":
            hidden_state = {
                "hidden_state": hidden_state[0].detach().cpu().numpy(),
            }
        else:
            hidden_state = None

        return hidden_state


registry.register_all(
    SequenceFn,
    {
        "LSTM": LSTMModel,
        "GRU": GRUModel,
    },
)

registry.register("SequenceModel", SequenceModel, SequenceModel)

get_sequence_fn = getattr(registry, f"get_{SequenceFn.type_name()}")
get_sequence_model = getattr(registry, f"get_{SequenceModel.type_name()}")
