import torch
from torch import nn

from hive.utils.registry import registry, Registrable


class SequenceModule(nn.Module, Registrable):
    """
    Base sequence neural network architecture.
    """

    def __init__(
        self,
        rnn_hidden_size=128,
        num_rnn_layers=1,
    ):
        """
        Args:
            rnn_hidden_size (int):  The number of features in the hidden state h.
            num_rnn_layers (int): Number of recurrent layers.
        """
        super().__init__()
        self._rnn_hidden_size = rnn_hidden_size
        self._num_rnn_layers = num_rnn_layers
        self.core = None

    def forward(self, x, hidden_state=None):
        x, hidden_state = self.core(x, hidden_state)
        return x, hidden_state

    @property
    def hidden_size(self):
        return self._rnn_hidden_size

    @classmethod
    def type_name(cls):
        return "sequence_fn"


class LSTMModule(SequenceModule):
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
            batch_first (bool): If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
        """
        super().__init__(
            rnn_hidden_size=rnn_hidden_size,
            num_rnn_layers=num_rnn_layers,
        )
        self.core = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=self._rnn_hidden_size,
            num_layers=self._num_rnn_layers,
            batch_first=batch_first,
        )

    def init_hidden(self, batch_size, device="cpu"):
        hidden_state = (
            torch.zeros(
                (self._num_rnn_layers, batch_size, self._rnn_hidden_size),
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros(
                (self._num_rnn_layers, batch_size, self._rnn_hidden_size),
                dtype=torch.float32,
                device=device,
            ),
        )

        return hidden_state


class GRUModule(SequenceModule):
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
            batch_first (bool): If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
        """
        super().__init__(
            rnn_hidden_size=rnn_hidden_size,
            num_rnn_layers=num_rnn_layers,
        )
        self.core = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=self._rnn_hidden_size,
            num_layers=self._num_rnn_layers,
            batch_first=batch_first,
        )

    def init_hidden(self, batch_size, device="cpu"):
        hidden_state = torch.zeros(
            (self._num_rnn_layers, batch_size, self._rnn_hidden_size),
            dtype=torch.float32,
            device=device,
        )

        return hidden_state


registry.register_all(
    SequenceModule,
    {
        "LSTM": LSTMModule,
        "GRU": GRUModule,
    },
)

get_sequence_fn = getattr(registry, f"get_{SequenceModule.type_name()}")
