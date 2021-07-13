import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from hive.agents.qnets.rainbow_mlp import NoisyLinear
from hive.agents.qnets.conv import conv2d_output_shape


class ComplexConv(nn.Module):
    """Convolution function approximator for Q-Learning."""

    def __init__(
        self,
        in_dim,
        out_dim,
        channels,
        mlp_layers,
        kernel_sizes=1,
        strides=1,
        paddings=0,
        normalization_factor=255,
        noisy=False,
        dueling=False,
        sigma_init=0.5,
        atoms=1,
    ):
        super(ComplexConv, self).__init__()
        self._normalization_factor = normalization_factor

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(channels)
        if isinstance(strides, int):
            strides = [strides] * len(channels)
        if isinstance(paddings, int):
            paddings = [paddings] * len(channels)

        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(strides)
        assert len(channels) == len(paddings)

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

        self._noisy = noisy
        self._dueling = dueling
        self._sigma_init = sigma_init
        self._in_dim = np.prod(in_dim)
        self._hidden_units = self.conv_out_size(h, w)
        if self._dueling:
            num_mlp_layers = max(len(mlp_layers) - 1, 2)
        self._num_mlp_layers = num_mlp_layers
        self._out_dim = out_dim
        self._atoms = atoms
        self.init_networks()

    def init_networks(self):

        if self._dueling:
            """In dueling, we have two heads - one for estimating advantage function and one for
            estimating value function. If `noisy` is also true, then each of these layers will
            be NoisyLinear()"""

            if self._noisy:

                self.output_layer_adv = nn.Sequential(
                    NoisyLinear(
                        self._hidden_units, self._hidden_units, self._sigma_init
                    ),
                    nn.ReLU(),
                    NoisyLinear(
                        self._hidden_units,
                        self._out_dim * self._atoms,
                        self._sigma_init,
                    ),
                )

                self.output_layer_val = nn.Sequential(
                    NoisyLinear(
                        self._hidden_units, self._hidden_units, self._sigma_init
                    ),
                    nn.ReLU(),
                    NoisyLinear(
                        self._hidden_units,
                        1 * self._atoms,
                        self._sigma_init,
                    ),
                )

            else:
                self.output_layer_adv = nn.Sequential(
                    nn.Linear(self._hidden_units, self._hidden_units, self._sigma_init),
                    nn.ReLU(),
                    nn.Linear(
                        self._hidden_units,
                        self._out_dim * self._atoms,
                        self._sigma_init,
                    ),
                )

                self.output_layer_val = nn.Sequential(
                    nn.Linear(self._hidden_units, self._hidden_units, self._sigma_init),
                    nn.ReLU(),
                    nn.Linear(
                        self._hidden_units,
                        1 * self._atoms,
                        self._sigma_init,
                    ),
                )
        else:
            if self._noisy:
                self.output_layer = NoisyLinear(
                    self._hidden_units, self._out_dim * self._atoms, self._sigma_init
                )
            else:
                self.output_layer = nn.Linear(
                    self._hidden_units, self._out_dim * self._atoms
                )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        elif len(x.shape) == 5:
            x = x.reshape(x.size(0), -1, x.size(-2), x.size(-1))
        x = x.type(torch.float)
        x = x / self._normalization_factor
        x = self.conv(x)
        x = torch.flatten(x, 1)

        if self._dueling:
            adv = self.output_layer_adv(x)
            val = self.output_layer_val(x)

            if len(adv.shape) == 1:
                x = val + adv - adv.mean(0)
            else:
                x = (
                    val
                    + adv
                    - adv.mean(1).unsqueeze(1).expand(x.shape[0], self._out_dim)
                )

        else:
            x = self.output_layer(x)

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


class DistributionalConv(ComplexConv):
    """Distributional Convolution function approximator for Q-Learning."""

    def __init__(
        self,
        in_dim,
        out_dim,
        supports,
        channels,
        mlp_layers,
        kernel_sizes=1,
        strides=1,
        paddings=0,
        normalization_factor=255,
        noisy=True,
        dueling=True,
        sigma_init=0.5,
        atoms=51,
    ):
        super().__init__(
            in_dim,
            out_dim,
            channels,
            mlp_layers,
            kernel_sizes,
            strides,
            paddings,
            normalization_factor,
            noisy,
            dueling,
            sigma_init,
            atoms,
        )
        self._supports = supports

    def forward(self, x):
        x = self.dist(x)
        x = torch.sum(x * self._supports, dim=2)
        return x

    def dist(self, x):

        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        elif len(x.shape) == 5:
            x = x.reshape(x.size(0), -1, x.size(-2), x.size(-1))
        x = x.type(torch.float)
        x = x / self._normalization_factor
        x = self.conv(x)
        x = torch.flatten(x, 1)

        if self._dueling:
            adv = self.output_layer_adv(x)
            adv = adv.view(-1, self._out_dim, self._atoms)
            val = self.output_layer_val(x)
            val = val.view(-1, 1, self._atoms)
            x = val + adv - adv.mean(dim=1, keepdim=True)

        else:
            x = self.output_layer(x)

        x = x.view(-1, self._out_dim, self._atoms)
        x = F.softmax(x, dim=-1)
        x = x.clamp(min=1e-3)

        return x
