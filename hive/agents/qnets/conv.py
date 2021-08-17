from hive.agents.qnets.mlp import MLPNetwork
import torch
from torch import nn


class ConvNetwork(nn.Module):
    """
    Simple convolutional network approximator for Q-Learning.
    """

    def __init__(
        self,
        in_dim,
        channels,
        mlp_layers,
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
                height)
            out_dim (int): The action dimension
            channels (list): The size of output channel for each convolutional layer
            mlp_layers (list): The size of neurons for each mlp layer after the
                convolutional layers
            kernel_sizes (list | int): The kernel size for each convolutional layer
            strides (list | int): The stride used for each convolutional layer
            paddings (list | int): The size of the padding used for each convolutional
                layer
            normalization_factor (float | int): What the input is divided by before
                the forward pass of the network
        """
        super().__init__()
        self._normalization_factor = normalization_factor

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(channels)
        if isinstance(strides, int):
            strides = [strides] * len(channels)
        if isinstance(paddings, int):
            paddings = [paddings] * len(channels)

        if not all(len(x) == len(channels) for x in [kernel_sizes, strides, paddings]):
            raise ValueError("The lengths of the parameter lists must be the same")

        c, h, w = in_dim
        # Convolutional Layers
        channels.insert(0, c)
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

        # MLP Layers
        self.head = torch.nn.Sequential(
            MLPNetwork(
                self.conv_out_size(h, w), mlp_layers, noisy=noisy, std_init=std_init
            ),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        elif len(x.shape) == 5:
            x = x.reshape(x.size(0), -1, x.size(-2), x.size(-1))
        x = x.float()
        x = x / self._normalization_factor
        x = self.conv(x)
        x = self.head(x)
        return x

    def conv_out_size(self, h, w):
        """
        Helper function ot return the output size for a given input shape,
        without actually performing a forward pass through the model.
        """
        for child in self.conv.children():
            try:
                h, w = conv2d_output_shape(
                    h, w, child.kernel_size, child.stride, child.padding
                )
            except AttributeError:
                pass
            try:
                c = child.out_channels
            except AttributeError:
                pass
        return h * w * c


def conv2d_output_shape(h, w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Returns output H, W after convolution/pooling on input H, W.
    """
    kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 2
    sh, sw = stride if isinstance(stride, tuple) else (stride,) * 2
    ph, pw = padding if isinstance(padding, tuple) else (padding,) * 2
    d = dilation
    h = (h + (2 * ph) - (d * (kh - 1)) - 1) // sh + 1
    w = (w + (2 * pw) - (d * (kw - 1)) - 1) // sw + 1
    return h, w
