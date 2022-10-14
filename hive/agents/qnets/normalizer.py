from typing import Tuple

import numpy as np
import torch


class NormalizationFn(torch.nn.Module):
    """Implements a normalization function. Transforms output by
    normalising the input data by the running :obj:`mean` and
    :obj:`std`, and clipping the normalised data on :obj:`clip`
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        epsilon: float = 1e-4,
        clip: np.float32 = np.inf,
    ):
        """
        Args:
            epsilon (float): minimum value of variance to avoid division by 0.
            shape (tuple[int]): The shape of input data.
            clip (np.float32): The clip value for the normalised data.
        """
        super().__init__()
        self.mean = np.zeros(shape, np.float32)
        self.std = np.ones(shape, np.float32)
        self._mean = torch.nn.Parameter(
            torch.as_tensor(self.mean, dtype=torch.float32), requires_grad=False
        )
        self._std = torch.nn.Parameter(
            torch.as_tensor(self.std, dtype=torch.float32), requires_grad=False
        )

        self.eps = epsilon
        self.shape = shape
        self.clip = clip
        self.size = 0

    def forward(self, val):
        with torch.no_grad():
            val = (val - self._mean) / self._std
            if self.clip is not None:
                val = torch.clamp(val, -self.clip, self.clip)
        return val

    def unnormalize(self, val):
        return val * self._std + self._mean

    def update(self, values):
        """
        Implementation of formula:
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """
        values = np.array(values).reshape(-1, *self.shape)
        batch_size = values.shape[0]
        batch_mean = values.mean(axis=0)
        batch_std = values.std(axis=0)

        delta = batch_mean - self.mean
        total_size = self.size + batch_size
        updated_mean = self.mean + delta * batch_size / total_size
        m_a = self.std**2 * self.size
        m_b = batch_std**2 * batch_size
        M2 = m_a + m_b + np.square(delta) * self.size * batch_size / total_size
        updated_std = np.maximum(np.sqrt(M2 / total_size), self.eps)

        self.mean = updated_mean
        self.std = updated_std
        self.size = total_size

        self._mean.data.copy_(torch.as_tensor(self.mean, dtype=torch.float32))
        self._std.data.copy_(torch.as_tensor(self.std, dtype=torch.float32))
