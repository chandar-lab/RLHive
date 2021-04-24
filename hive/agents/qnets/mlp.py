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


class ComplexMLP(nn.Module):
    """Simple MLP function approximator for Q-Learning."""

    def __init__(self, in_dim, out_dim, supports, hidden_units=256, num_hidden_layers=1, noisy=True, dueling=True, sigma_init=0.5):
        super().__init__()

        self._noisy = noisy
        self._dueling = dueling
        self._sigma_init = sigma_init
        self._in_dim = in_dim
        self._out_dim = out_dim

        self.input_layer = nn.Sequential(
            nn.Linear(self._in_dim, hidden_units), nn.ReLU()
        )
        self.hidden_layers = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(hidden_units, hidden_units), nn.ReLU())
                for _ in range(num_hidden_layers - 1)
            ]
        )
        self.output_layer = nn.Linear(hidden_units, self._out_dim)

        if self._dueling:

            if self._noisy:

                self.fc1 = NoisyLinear(self._in_dim, hidden_units, self._sigma_init)
                self.fc1_adv = NoisyLinear(hidden_units, hidden_units, self._sigma_init)
                self.fc2_adv = NoisyLinear(hidden_units, self._out_dim, self._sigma_init)

                self.fc1_val = NoisyLinear(hidden_units, hidden_units, self._sigma_init)
                self.fc2_val = NoisyLinear(hidden_units, 1, self._sigma_init)

            else:

                self.fc1_adv = nn.Linear(hidden_units, hidden_units)
                self.fc2_adv = nn.Linear(hidden_units, self._out_dim)

                self.fc1_val = nn.Linear(hidden_units, hidden_units)
                self.fc2_val = nn.Linear(hidden_units, 1)

        else:

            if self._noisy:
                self.fc1 = NoisyLinear(self._in_dim, hidden_units, self._sigma_init)
                self.fc2 = NoisyLinear(hidden_units, self._out_dim, self._sigma_init)

        self.relu = nn.ReLU()

    def forward(self, x):
        if self._dueling:
            if self._noisy:
                x = F.relu(self.fc1(x))
            else:
                x = self.input_layer(x)
            adv = self.relu(self.fc1_adv(x))
            adv = self.fc2_adv(adv)

            val = self.relu(self.fc1_val(x))
            val = self.fc2_val(val)

            if len(adv.shape) == 1:
                x = val + adv - adv.mean(0)
            else:
                x = val + adv - adv.mean(1).unsqueeze(1).expand(x.shape[0], self._out_dim)

        else:
            if self._noisy:
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
            else:
                x = self.input_layer(x)
                x = self.hidden_layers(x)
                x = self.output_layer(x)

        return x

    def sample_noise(self):
        if self._dueling:
            self.fc1.sample_noise()
            self.fc1_adv.sample_noise()
            self.fc2_adv.sample_noise()
            self.fc1_val.sample_noise()
            self.fc2_val.sample_noise()
        else:
            self.fc1.sample_noise()
            self.fc2.sample_noise()


class DistributionalMLP(nn.Module):
    """Simple MLP function approximator for Q-Learning."""

    def __init__(self, in_dim, out_dim, supports, hidden_units=256, num_hidden_layers=1,
                 noisy=True, dueling=True, sigma_init=0.5, atoms=51):
        super().__init__()

        self._noisy = noisy
        self._dueling = dueling
        self._sigma_init = sigma_init
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._atoms = atoms
        self._supports = supports

        self.input_layer = nn.Sequential(
            nn.Linear(self._in_dim, hidden_units), nn.ReLU()
        )
        self.hidden_layers = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(hidden_units, hidden_units), nn.ReLU())
                for _ in range(num_hidden_layers - 1)
            ]
        )
        self.output_layer = nn.Linear(hidden_units, self._out_dim * self._atoms)

        if self._dueling:

            if self._noisy:

                self.fc1 = NoisyLinear(self._in_dim, hidden_units, self._sigma_init)
                self.fc1_adv = NoisyLinear(hidden_units, hidden_units, self._sigma_init)
                self.fc2_adv = NoisyLinear(hidden_units, self._out_dim * self._atoms, self._sigma_init)

                self.fc1_val = NoisyLinear(hidden_units, hidden_units, self._sigma_init)
                self.fc2_val = NoisyLinear(hidden_units, 1 * self._atoms, self._sigma_init)

            else:

                self.fc1_adv = nn.Linear(hidden_units, hidden_units)
                self.fc2_adv = nn.Linear(hidden_units, self._out_dim * self._atoms)

                self.fc1_val = nn.Linear(hidden_units, hidden_units)
                self.fc2_val = nn.Linear(hidden_units, 1 * self._atoms)

        else:

            if self._noisy:
                self.fc1 = NoisyLinear(self._in_dim, hidden_units, self._sigma_init)
                self.fc2 = NoisyLinear(hidden_units, self._out_dim * self._atoms, self._sigma_init)

        self.relu = nn.ReLU()

    def forward(self, x):
        if self._dueling:
            if self._noisy:
                x = F.relu(self.fc1(x))
            else:
                x = self.input_layer(x)
            adv = self.relu(self.fc1_adv(x))
            adv = self.fc2_adv(adv).view(-1, self._out_dim, self._atoms)

            val = self.relu(self.fc1_val(x))
            val = self.fc2_val(val).view(-1, 1, self._atoms)

            x = val + adv - adv.mean(dim=1).view(-1, 1, self._atoms)

        else:
            x = self.dist(x)

        x = torch.sum(x * self._supports, dim=2)
        return x

    def sample_noise(self):
        if self._dueling:
            self.fc1.sample_noise()
            self.fc1_adv.sample_noise()
            self.fc2_adv.sample_noise()
            self.fc1_val.sample_noise()
            self.fc2_val.sample_noise()
        else:
            self.fc1.sample_noise()
            self.fc2.sample_noise()

    def dist(self, x):

        if self._noisy:
            x = F.relu(self.fc1(x))
        else:
            x = self.input_layer(x)

        if self._dueling:
            adv = self.relu(self.fc1_adv(x))
            adv = self.fc2_adv(adv).view(-1, self._out_dim, self._atoms)

            val = self.relu(self.fc1_val(x))
            val = self.fc2_val(val).view(-1, 1, self._atoms)

            x = val + adv - adv.mean(dim=1).view(-1, 1, self._atoms)

        else:
            if self._noisy:
                x = self.fc2(x)
            else:
                x = self.hidden_layers(x)
                x = self.output_layer(x)

        x = x.view(-1, self._out_dim, self._atoms)
        x = F.softmax(x, dim=-1)
        x = x.clamp(min=1e-3)

        return x