import numpy as np
import torch
from torch import nn

from hive.agents.qnets.mlp import MLPNetwork
from hive.agents.qnets.conv import ConvNetwork
from hive.agents.qnets.utils import calculate_output_dim
from hive.agents.qnets.sequence_models import SequenceFn


class ConvRNNNetwork(nn.Module):
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
        sequence_fn: SequenceFn,
        channels=None,
        mlp_layers=None,
        kernel_sizes=1,
        strides=1,
        paddings=0,
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
            channels (list): The size of output channel for each convolutional layer.
            mlp_layers (list): The number of neurons for each mlp layer after the
                convolutional layers.
            kernel_sizes (list | int): The kernel size for each convolutional layer
            strides (list | int): The stride used for each convolutional layer.
            paddings (list | int): The size of the padding used for each convolutional
                layer.
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
        if channels is not None:
            if isinstance(kernel_sizes, int):
                kernel_sizes = [kernel_sizes] * len(channels)
            if isinstance(strides, int):
                strides = [strides] * len(channels)
            if isinstance(paddings, int):
                paddings = [paddings] * len(channels)

            if not all(
                len(x) == len(channels) for x in [kernel_sizes, strides, paddings]
            ):
                raise ValueError("The lengths of the parameter lists must be the same")

            # Convolutional Layers
            channels.insert(0, in_dim[0])
            conv_seq = []
            for i in range(0, len(channels) - 1):
                conv_seq.append(
                    torch.nn.Conv2d(
                        in_channels=channels[i],
                        out_channels=channels[i + 1],
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        padding=paddings[i],
                    )
                )
                conv_seq.append(torch.nn.ReLU())
            self.conv = torch.nn.Sequential(*conv_seq)
        else:
            self.conv = torch.nn.Identity()

        # RNN Layers
        conv_output_size = calculate_output_dim(self.conv, in_dim)
        self.rnn = sequence_fn(
            rnn_input_size=np.prod(conv_output_size),
        )

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

    def forward(self, x, hidden_state=None):
        if len(x.shape) == 4:
            # Act. Sequence length is 1.
            B, C, H, W = x.size()
            L = 1
        elif len(x.shape) == 5:
            # Update. Sequence length pre-defined.
            B, L, C, H, W = x.size()
        x = x.reshape(B * L, C, H, W)

        x = x.float()
        x = x / self._normalization_factor
        x = self.conv(x)

        _, C, H, W = x.size()
        x = x.view(B, L, C, H, W)
        if hidden_state is None:
            hidden_state = self.init_hidden(B)
        x = torch.flatten(x, start_dim=2, end_dim=-1)  # (B, L, -1)
        x, hidden_state = self.rnn(x, hidden_state)
        x = self.mlp(x.reshape((B * L, -1)))
        return x, hidden_state

    def init_hidden(self, batch_size):
        hidden_state = self.rnn.init_hidden(
            batch_size=batch_size,
        )

        return hidden_state
