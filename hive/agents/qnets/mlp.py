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


class DuelingMLP(nn.Module):
    """Simple MLP function approximator for Q-Learning."""

    def __init__(self, in_dim, out_dim, hidden_units=256, num_hidden_layers=1):
        super().__init__()
        self.out_dim = out_dim
        self.input_layer = nn.Sequential(nn.Linear(in_dim[0], hidden_units), nn.ReLU())
        self.hidden_layers = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(hidden_units, hidden_units), nn.ReLU())
                for _ in range(num_hidden_layers - 1)
            ]
        )
        self.fc1_adv = nn.Linear(hidden_units, hidden_units)
        self.fc2_adv = nn.Linear(hidden_units, out_dim)

        self.fc1_val = nn.Linear(hidden_units, hidden_units)
        self.fc2_val = nn.Linear(hidden_units, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)

        adv = self.relu(self.fc1_adv(x))
        adv = self.fc2_adv(adv)

        val = self.relu(self.fc1_val(x))
        val = self.fc2_val(val)

        # print("val shape = ", val.shape)
        # print("adv shape = ", adv.shape)
        # print("adv mean shape = ", adv.mean(1).shape)
        if len(adv.shape) == 1:
            print("val shape = ", val.shape)
            print("adv shape = ", adv.shape)
            print("adv mean shape = ", adv.mean(0).shape)
            print("val = ", val)
            print("adv = ", adv)
            print("mean = ", adv.mean(0))
            x = val + adv - adv.mean(0)
            print("x shape = ", x.shape)
            print("x = ", x)
        else:
            x = val + adv - adv.mean(1).unsqueeze(1).expand(x.shape[0], self.out_dim)
        return x
