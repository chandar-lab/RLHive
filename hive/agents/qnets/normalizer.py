from typing import Tuple

import numpy as np

from hive.utils.registry import Registrable, registry


# taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
class MeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = self.update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    def update_mean_var_count_from_moments(
        self, mean, var, count, batch_mean, batch_var, batch_count
    ):
        """Updates the mean, var and count using the previous mean, var, count
        and batch values."""
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count


class Normalizer(Registrable):
    """A wrapper for callables that produce normalization functions.

    These wrapped callables can be partially initialized through configuration
    files or command line arguments.
    """

    @classmethod
    def type_name(cls):
        """
        Returns:
            "norm_fn"
        """
        return "norm_fn"


class MovingAvgNormalizer(Normalizer):
    """Implements a moving average normalization and clipping function. Normalizes
    input data with the running mean and std. The normalized data is then clipped
    within the specified range.
    """

    def __init__(
        self, shape: Tuple[int, ...], epsilon: float = 1e-4, clip: np.float32 = np.inf
    ):
        """
        Args:
            epsilon (float): minimum value of variance to avoid division by 0.
            shape (tuple[int]): The shape of input data.
            clip (np.float32): The clip value for the normalised data.
        """
        super().__init__()
        self._rms = MeanStd(epsilon, shape)
        self._shape = shape
        self._epsilon = epsilon
        self._clip = clip

    def __call__(self, input_data):
        input_data = np.array([input_data])
        input_data = (
            (input_data - self._rms.mean) / np.sqrt(self._rms.var + self._epsilon)
        )[0]
        if self._clip is not None:
            input_data = np.clip(input_data, -self._clip, self._clip)
        return input_data

    def update(self, input_data):
        self._rms.update(input_data)


class RewardNormalizer(Normalizer):
    """Normalizes and clips rewards from the environment. Applies a discount-based
    scaling scheme, where the rewards are divided by the standard deviation of a
    rolling discounted sum of the rewards. The scaled rewards are then clipped within
    specified range.
    """

    def __init__(self, gamma: float, epsilon: float = 1e-4, clip: np.float32 = np.inf):
        """
        Args:
            gamma (float): discount factor for the agent.
            epsilon (float): minimum value of variance to avoid division by 0.
            clip (np.float32): The clip value for the normalised data.
        """
        super().__init__()
        self._return_rms = MeanStd(epsilon, ())
        self._epsilon = epsilon
        self._clip = clip
        self._gamma = gamma
        self._returns = np.zeros(1)

    def __call__(self, rew):
        rew = np.array([rew])
        rew = (rew / np.sqrt(self._return_rms.var + self._epsilon))[0]
        if self._clip is not None:
            rew = np.clip(rew, -self._clip, self._clip)
        return rew

    def update(self, rew, done):
        self._returns = self._returns * self._gamma + rew
        self._return_rms.update(self._returns)
        self._returns *= 1 - done


registry.register_all(
    Normalizer,
    {
        "RewardNormalizer": RewardNormalizer,
        "MovingAvgNormalizer": MovingAvgNormalizer,
    },
)

get_norm_fn = getattr(registry, f"get_{Normalizer.type_name()}")
