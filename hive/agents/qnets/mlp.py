import torch
from torch import nn
import torch.nn.functional as F
import math


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
    """Dueling MLP function approximator for Q-Learning."""

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

        if len(adv.shape) == 1:
            x = val + adv - adv.mean(0)
        else:
            x = val + adv - adv.mean(1).unsqueeze(1).expand(x.shape[0], self.out_dim)

        return x


class DistributionalMLP(nn.Module):
    """Simple distributionalMLP function approximator for Q-Learning."""

    def __init__(self, in_dim, out_dim, hidden_units=256, num_hidden_layers=1, num_atoms=10):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(in_dim[0], hidden_units), nn.ReLU())
        self.hidden_layers = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(hidden_units, hidden_units), nn.ReLU())
                for _ in range(num_hidden_layers - 1)
            ]
        )
        self.output_layer = nn.Linear(hidden_units, out_dim)
        self.out_dim = out_dim
        self.num_atoms = num_atoms

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        logits = x.view(batch_size, self.out_dim, self.num_atoms)
        probs = nn.functional.softmax(logits, 2)
        return probs


class NoisyLinear(nn.Module):
    def __init__(self, in_dim, out_dim, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_dim
        self.out_features = out_dim
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_dim, in_dim))
        self.weight_sigma = nn.Parameter(torch.empty(out_dim, in_dim))
        self.register_buffer('weight_epsilon', torch.empty(out_dim, in_dim))
        self.bias_mu = nn.Parameter(torch.empty(out_dim))
        self.bias_sigma = nn.Parameter(torch.empty(out_dim))
        self.register_buffer('bias_epsilon', torch.empty(out_dim))
        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def sample_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, inp):
        if self.training:
            return F.linear(inp, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(inp, self.weight_mu, self.bias_mu)


class NoisyMLP(nn.Module):
    # ref: https://github.com/qfettes/DeepRL-Tutorials/blob/master/05.DQN-NoisyNets.ipynb
    def __init__(self, in_dim, out_dim, hidden_units, num_hidden_layers, sigma_init=0.5):
        super(NoisyMLP, self).__init__()

        self.input_shape = in_dim
        self.num_actions = out_dim

        self.fc1 = NoisyLinear(self.input_shape[0], 512, sigma_init)
        self.fc2 = NoisyLinear(512, self.num_actions, sigma_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def sample_noise(self):
        self.fc1.sample_noise()
        self.fc2.sample_noise()