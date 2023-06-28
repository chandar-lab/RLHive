from typing import Protocol, Tuple, runtime_checkable

import numpy as np
from numba import njit

from hive.utils.registry import registry


@runtime_checkable
class AdvantageComputationFn(Protocol):
    """A wrapper for callables that produce Advantage Computation Functions."""

    def __call__(
        self,
        values: np.ndarray,
        last_values: np.ndarray,
        terminated: np.ndarray,
        dones: np.ndarray,
        rewards: np.ndarray,
        gamma: float,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...


@njit
def compute_gae_advantages(
    values: np.ndarray,
    last_values: np.ndarray,
    terminated: np.ndarray,
    dones: np.ndarray,
    rewards: np.ndarray,
    gamma: float,
    gae_lambda: float,
):
    """Helper function that computes advantages and returns using Generalized Advantage Estimation.

    Args:
        values (np.ndarray): Value estimates for each step. Should be of shape (T, E),
            where E is the number of environments and T is the number of steps.
        last_values (np.ndarray): Value estimate for the last step. Should be of shape
            (E,).
        terminated (np.ndarray): Terminated flags for each step. Should be of shape
            (T, E).
        dones (np.ndarray): Done flags for each step. Should be of shape (T, E).
        rewards (np.ndarray): Rewards for each step. Should be of shape (T, E).
        gamma (float): Discount factor.
        gae_lambda (float): GAE lambda parameter.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of advantages and returns. Both should be
            of shape (T, E).
    """
    last_gae_lambda = np.zeros((values.shape[1],))
    num_steps = len(values)
    advantages = np.zeros_like(rewards)
    values = np.concatenate((values, np.expand_dims(last_values, axis=0)), axis=0)
    for t in range(num_steps - 1, -1, -1):
        next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * (1.0 - terminated[t]) - values[t]
        last_gae_lambda = (
            delta + gamma * gae_lambda * (1.0 - dones[t]) * last_gae_lambda
        )
        advantages[t] = last_gae_lambda
    returns = advantages + values[:-1]
    return advantages, returns


def compute_standard_advantages(
    values: np.ndarray,
    last_values: np.ndarray,
    terminated: np.ndarray,
    dones: np.ndarray,
    rewards: np.ndarray,
    gamma: float,
    n_step: int = 1,
):
    """Helper function that computes advantages and returns using standard n-step
    returns and advantage estimation.

    Args:
        values (np.ndarray): Value estimates for each step. Should be of shape (T, E),
            where E is the number of environments and T is the number of steps.
        last_values (np.ndarray): Value estimate for the last step. Should be of shape
            (E,).
        terminated (np.ndarray): Terminated flags for each step. Should be of shape
            (T, E).
        dones (np.ndarray): Done flags for each step. Should be of shape (T, E).
        rewards (np.ndarray): Rewards for each step. Should be of shape (T, E).
        gamma (float): Discount factor.
        n_step (int): Number of steps to use for bootstrapping.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of advantages and returns. Both should be
            of shape (E, T).
    """
    idx = np.expand_dims(np.arange(values.shape[0]), 1)  # (T, 1)
    idx = np.broadcast_to(idx, values.shape)  # (T, E)
    terminated = np.concatenate(  # (T + n_step - 1, E)
        (terminated, np.zeros_like(terminated[: n_step - 1])), axis=0
    )
    dones = np.concatenate(  # (T + n_step - 1, E)
        (dones, np.ones_like(dones[: n_step - 1])), axis=0
    )

    # Calculate trajectory lengths
    dones = np.minimum(
        np.lib.stride_tricks.sliding_window_view(  # (T, E, n_step)
            dones, n_step, axis=0
        ).cumsum(axis=2),
        1,
    )
    trajectory_length = (n_step - dones.sum(axis=2)).astype(np.int64)  # (T, E)

    # Get next values
    last_values = np.tile(
        np.expand_dims(last_values, axis=0), (n_step, 1)
    )  # E -> (n_step, E)
    next_values = np.concatenate((values, last_values), axis=0)[
        idx + trajectory_length,
        np.expand_dims(np.arange(values.shape[1]), 0),
    ]  # (T, E)

    rewards = np.concatenate(  # (T + n_step - 1, E)
        (rewards, np.zeros_like(rewards[: n_step - 1])), axis=0
    )
    rewards = np.lib.stride_tricks.sliding_window_view(  # (T, E, n_step)
        rewards, n_step, axis=0
    )

    discount = np.expand_dims(gamma ** np.arange(n_step), (0, 1)) * (1 - dones)
    terminated = np.lib.stride_tricks.sliding_window_view(  # (T, E, n_step)
        terminated, n_step, axis=0
    )
    returns = (rewards * discount).sum(axis=2) + next_values * np.any(
        terminated, axis=2
    )

    advantages = returns - values
    return advantages, returns


registry.register_all_with_type(
    AdvantageComputationFn,
    {
        "gae_advantages": compute_gae_advantages,
        "standard_advantages": compute_standard_advantages,
    },
)
