from hive.agents.qnets.conv import ConvNetwork


class NatureAtariDQNModel(ConvNetwork):
    """The convolutional network used to train the DQN agent in the original
    Nature paper: https://www.nature.com/articles/nature14236
    """

    def __init__(self, in_dim):
        """
        Args:
            in_dim (tuple): The tuple of observations dimension
                (channels, width, height).
        """
        super().__init__(
            in_dim=in_dim,
            channels=[32, 64, 64],
            kernel_sizes=[8, 4, 3],
            strides=[4, 2, 1],
            paddings=[0, 1, 1],
            mlp_layers=[512],
        )
