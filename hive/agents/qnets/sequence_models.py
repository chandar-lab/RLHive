import torch
from torch import nn

from hive.utils.registry import registry, Registrable
from hive.agents.qnets.base import FunctionApproximator


class SequenceFn(Registrable):
    """A wrapper for callables that produce sequence functions."""

    @classmethod
    def type_name(cls):
        return "sequence_fn"


class SequenceModel(nn.Module):
    """
    Base sequence neural network architecture.
    """

    def __init__(
        self,
        rnn_hidden_size=128,
        num_rnn_layers=1,
        device="cpu",
    ):
        """
        Args:
            rnn_hidden_size (int):  The number of features in the hidden state h.
            num_rnn_layers (int): Number of recurrent layers.
            device: Device on which all computations should be run.
        """
        super().__init__()
        self._rnn_hidden_size = rnn_hidden_size
        self._num_rnn_layers = num_rnn_layers
        self.core = None
        self._device = device

    def forward(self, x, hidden_state=None):
        x, hidden_state = self.core(x, hidden_state)
        return x, hidden_state

    def update_device(self):
        self._device = next(self.core.parameters()).device


class LSTMModel(SequenceModel):
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
        super().__init__(
            rnn_hidden_size=rnn_hidden_size,
            num_rnn_layers=num_rnn_layers,
            device=device,
        )
        self.core = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=self._rnn_hidden_size,
            num_layers=self._num_rnn_layers,
            batch_first=batch_first,
        )

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


class GRUModel(SequenceModel):
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
        super().__init__(
            rnn_hidden_size=rnn_hidden_size,
            num_rnn_layers=num_rnn_layers,
            device=device,
        )
        self.core = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=self._rnn_hidden_size,
            num_layers=self._num_rnn_layers,
            batch_first=batch_first,
        )

    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(
            (self._num_rnn_layers, batch_size, self._rnn_hidden_size),
            dtype=torch.float32,
            device=self._device,
        )

        return hidden_state


registry.register_all(
    SequenceFn,
    {
        "LSTM": LSTMModel,
        "GRU": GRUModel,
    },
)

get_sequence_fn = getattr(registry, f"get_{SequenceFn.type_name()}")
