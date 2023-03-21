import numpy as np

from hive.utils.registry import registry
from hive.utils.utils import Registrable


class AdvantageComputationFn(Registrable):
    """A wrapper for callables that produce Advantage Computation Functions."""

    @classmethod
    def type_name(cls):
        """
        Returns:
            "advantage_computation_fn"
        """
        return "advantage_computation_fn"


def compute_gae_advantages(
    values: np.ndarray,
    last_values: np.ndarray,
    dones: np.ndarray,
    rewards: np.ndarray,
    gamma: float,
    gae_lambda: float,
):
    """Helper function that computes advantages and returns using Generalized Advantage Estimation.

    Args:
        values (np.ndarray): Value estimates for each step.
        last_values (np.ndarray): Value estimate for the last step.
        dones (np.ndarray): Done flags for each step.
        rewards (np.ndarray): Rewards for each step.
        gamma (float): Discount factor.
        gae_lambda (float): GAE lambda parameter.
    """
    last_gae_lambda = 0
    num_steps = len(values)
    advantages = np.zeros_like(rewards)
    for t in reversed(range(num_steps)):
        next_values = last_values if t == num_steps - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        advantages[t] = last_gae_lambda = (
            delta + gamma * gae_lambda * next_non_terminal * last_gae_lambda
        )
    returns = advantages + values
    return advantages, returns


def compute_standard_advantages(
    values: np.ndarray,
    last_values: np.ndarray,
    dones: np.ndarray,
    rewards: np.ndarray,
    gamma: float,
):
    """Helper function that computes advantages and returns using standard advantage estimation.

    Args:
        values (np.ndarray): Value estimates for each step.
        last_values (np.ndarray): Value estimate for the last step.
        dones (np.ndarray): Done flags for each step.
        rewards (np.ndarray): Rewards for each step.
        gamma (float): Discount factor.
    """
    num_steps = len(values)
    advantages = np.zeros_like(rewards)
    returns = np.zeros_like(rewards)
    for t in reversed(range(num_steps)):
        next_return = last_values if t == num_steps - 1 else returns[t + 1]
        next_non_terminal = 1.0 - dones[t]
        returns[t] = rewards[t] + gamma * next_non_terminal * next_return
    advantages = returns - values
    return advantages, returns


registry.register_all(
    AdvantageComputationFn,
    {
        "gae_advantages": compute_gae_advantages,
        "standard_advantages": compute_standard_advantages,
    },
)
