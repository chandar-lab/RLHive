import torch
from torch import nn
from hive.agents.qnets.utils import conv2d_output_shape


class NatureDQNModel(nn.Module):
    """
        The convolutional network used to train the DQN agent.
    """

    def __init__(self, in_dim, out_dim):
        """
        Args:
            in_dim (tuple): The tuple of observations dimension (channels, width, height)
            out_dim (int): The action dimension
        """
        super.__init__()

        c, h, w = in_dim

        # Default Convolutional Layers
        channels = [c, 32, 64, 64]
        kernel_sizes = [8, 4, 3]
        strides = [4, 2, 1]
        paddings = [0, 1, 1]

        conv_layers = [
            torch.nn.Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=ks,
                stride=s,
                padding=p
            )
            for (in_c, out_c, ks, s, p) in
            zip(channels[:-1], channels[1:], kernel_sizes, strides, paddings)
        ]
        conv_seq = list()
        for conv_layer in conv_layers:
            conv_seq.extend([conv_layer, torch.nn.ReLU])
        self.conv = torch.nn.Sequential(*conv_seq)

        # Default MLP Layers
        conv_out_size = self.conv.conv_out_size(h, w)
        head_units = [conv_out_size, 512, out_dim]
        head_layers = [
            torch.nn.Linear(i, o) for i, o in zip(head_units[:-1], head_units[1:])
        ]
        head_seq = list()
        for head_layer in head_layers:
            head_seq.extend([head_layer, torch.nn.ReLU])
        self.head = torch.nn.Sequential(*head_seq)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        batch_size = x.size(0)

        x = x.type(torch.float)
        x = x.mul_(1. / 255)
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
                h, w = conv2d_output_shape(h, w, child.kernel_size,
                                           child.stride, child.padding)
            except AttributeError:
                pass
            try:
                c = child.out_channels
            except AttributeError:
                pass
        return h * w * c
