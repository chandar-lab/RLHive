from hive.agents.qnets.conv import SimpleConvModel


class MinAtarDQNModel(SimpleConvModel):
    """
    The convolutional network used to train the DQN agent.
    """

    def __init__(self, in_dim, out_dim):
        """
        Args:
            in_dim (tuple): The tuple of observations dimension (channels, width, height)
            out_dim (int): The action dimension
        """
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            channels=[16],
            kernel_sizes=[3],
            strides=[1],
            paddings=[0],
            mlp_layers=[128],
        )
