from flax import linen as nn


class JaxDQNNetwork(nn.Module):
    """Implements the standard DQN value computation. Transforms output from
    :obj:`base_network` with output dimension :obj:`hidden_dim` to dimension
    :obj:`out_dim`, which should be equal to the number of actions.
    """

    def __init__(
        self,
        base_network: nn.Module,
        hidden_dim: int,
        out_dim: int,
        linear_fn: nn.Module = None,
    ):
        """
        Args:
            base_network (torch.nn.Module): Backbone network that computes the
                representations that are used to compute action values.
            hidden_dim (int): Dimension of the output of the :obj:`network`.
            out_dim (int): Output dimension of the DQN. Should be equal to the
                number of actions that you are computing values for.
            linear_fn (torch.nn.Module): Function that will create the
                :py:class:`torch.nn.Module` that will take the output of
                :obj:`network` and produce the final action values. If
                :obj:`None`, a :py:class:`torch.nn.Linear` layer will be used.
        """
        super().__init__()
        self.base_network = base_network
        self._linear_fn = linear_fn if linear_fn is not None else nn.linear
        self.output_layer = self._linear_fn(hidden_dim, out_dim)

    def forward(self, x):
        x = self.base_network(x)
        x = x.flatten(start_dim=1)
        return self.output_layer(x)


## class DuelingNetwork(nn.Module): TODO add this class


## class DistributionalNetwork(nn.Module): TODO add this class
