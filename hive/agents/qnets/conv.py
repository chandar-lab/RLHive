import torch
from torch import nn

from hive.agents.qnets.mlp import MLPNetwork
from hive.agents.qnets.utils import calculate_output_dim


class ConvNetwork(nn.Module):
    """
    Basic convolutional neural network architecture. Applies a number of
    convolutional layers (each followed by a ReLU activation), and then
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

        if mlp_layers is not None:
            # MLP Layers
            conv_output_size = calculate_output_dim(self.conv, in_dim)
            self.mlp = MLPNetwork(
                conv_output_size, mlp_layers, noisy=noisy, std_init=std_init
            )
        else:
            self.mlp = torch.nn.Identity()

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        elif len(x.shape) == 5:
            x = x.reshape(x.size(0), -1, x.size(-2), x.size(-1))
        x = x.float()
        x = x / self._normalization_factor
        x = self.conv(x)
        x = self.mlp(x)
        return x
