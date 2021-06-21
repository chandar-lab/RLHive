import torch
from torch import nn

from hive.agents.qnets.utils import conv2d_output_shape


class SimpleConvModel(nn.Module):
    """
    Simple convolutional network approximator for Q-Learning.
    """

    def __init__(
        self, in_dim, out_dim, channels, kernel_sizes, strides, paddings, mlp_layers
    ):
        """
        Args:
            in_dim (tuple): The tuple of observations dimension (channels, width, height)
            out_dim (int): The action dimension
            channels (list): The size of output channel for each convolutional layer
            kernel_sizes (list): The kernel size for each convolutional layer
            strides (list): The stride used for each convolutional layer
            paddings (list): The size of the padding used for each convolutional layer
            mlp_layers (list): The size of neurons for each mlp layer after the convolutional layers
        """

        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(strides)
        assert len(channels) == len(paddings)

        super().__init__()

        print("in_dim = ", in_dim)
        c, h, w = in_dim

        # Default Convolutional Layers
        channels = [c] + channels

        conv_layers = [
            torch.nn.Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=ks,
                stride=s,
                padding=p,
            )
            for (in_c, out_c, ks, s, p) in zip(
                channels[:-1], channels[1:], kernel_sizes, strides, paddings
            )
        ]
        conv_seq = list()
        for conv_layer in conv_layers:
            conv_seq.extend([conv_layer, torch.nn.ReLU()])
        self.conv = torch.nn.Sequential(*conv_seq)

        # Default MLP Layers
        conv_out_size = self.conv_out_size(h, w)
        head_units = [conv_out_size] + mlp_layers + [out_dim]
        head_layers = [
            torch.nn.Linear(i, o) for i, o in zip(head_units[:-1], head_units[1:])
        ]
        head_seq = list()
        for head_layer in head_layers[:-1]:
            head_seq.extend([head_layer, torch.nn.ReLU()])
        head_seq.extend([head_layers[-1]])
        self.head = torch.nn.Sequential(*head_seq)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        batch_size = x.size(0)

        x = x.type(torch.float)
        x = x.mul_(1.0 / 255)
        conv_out = self.conv(x)
        q = self.head(conv_out.view(batch_size, -1))
        return q

    def conv_out_size(self, h, w, c=None):
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
