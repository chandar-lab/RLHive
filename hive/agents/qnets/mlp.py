from torch import nn


class SimpleMLP(nn.Module):
    """Simple MLP function approximator for Q-Learning."""

    def __init__(self, in_dim, out_dim, hidden_units=256, num_hidden_layers=1):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(in_dim[0], hidden_units), nn.ReLU())
        self.hidden_layers = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(hidden_units, hidden_units), nn.ReLU())
                for _ in range(num_hidden_layers - 1)
            ]
        )
        self.output_layer = nn.Linear(hidden_units, out_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        return self.output_layer(x)
