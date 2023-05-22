import numpy as np
import torch
from torch import nn

from hive.utils.registry import registry, Registrable

from hive.agents.qnets.mlp import MLPNetwork
from hive.agents.qnets.utils import calculate_output_dim


class SequenceFunctionApproximator(Registrable):
    """A wrapper for callables that produce sequence functions."""

    @classmethod
    def type_name(cls):
        return "SequenceFunctionApproximator"


class LSTMModel(nn.Module):
    """
    A multi-layer long short-term memory (LSTM) RNN.
    """

    def __init__(
        self,
        rnn_input_size=256,
        rnn_hidden_size=128,
        num_rnn_layers=1,
        batch_first=True,
        device="cpu",
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
        self._device = device
        self.core = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=self._rnn_hidden_size,
            num_layers=self._num_rnn_layers,
            batch_first=batch_first,
        )

    def forward(self, x, hidden_state=None):
        x, hidden_state = self.core(x, hidden_state)
        return x, hidden_state

    def _update_device(self):
        self._device = next(self.core.parameters()).device

    def init_hidden(self, batch_size):
        hidden_state = (
            torch.zeros(
                (self._num_rnn_layers, batch_size, self._rnn_hidden_size),
                dtype=torch.float32,
                device=self._device,
            ),
            torch.zeros(
                (self._num_rnn_layers, batch_size, self._rnn_hidden_size),
                dtype=torch.float32,
                device=self._device,
            ),
        )

        return hidden_state

    def update_device(self):
        self._update_device()


class GRUModel(nn.Module):
    """
    A multi-layer gated recurrent unit (GRU) RNN.
    """

    def __init__(
        self,
        rnn_input_size=256,
        rnn_hidden_size=128,
        num_rnn_layers=1,
        batch_first=True,
        device="cpu",
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
        self._device = device
        self.core = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=self._rnn_hidden_size,
            num_layers=self._num_rnn_layers,
            batch_first=batch_first,
        )

    def forward(self, x, hidden_state=None):
        x, hidden_state = self.core(x, hidden_state)
        return x, hidden_state

    def _update_device(self):
        self._device = next(self.core.parameters()).device

    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(
            (self._num_rnn_layers, batch_size, self._rnn_hidden_size),
            dtype=torch.float32,
            device=self._device,
        )

        return hidden_state

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
        sequence_fn: SequenceFunctionApproximator,
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
    SequenceFunctionApproximator,
    {
        "LSTM": LSTMModel,
        "GRU": GRUModel,
    },
)

get_sequence_fn = getattr(registry, f"get_{SequenceFunctionApproximator.type_name()}")
