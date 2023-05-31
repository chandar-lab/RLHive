import numpy as np
import torch


def get_stacked_state(observation, observation_stack, stack_size):
    """Create a stacked state for the agent. The previous observations recorded
    by this agent are stacked with the current observation. If not enough
    observations have been recorded, zero arrays are appended.

    Args:
        observation: Current observation.
    """

    if stack_size == 1:
        return observation, observation_stack
    while len(observation_stack) < stack_size - 1:
        observation_stack.append(zeros_like(observation))

    stacked_observation = concatenate(list(observation_stack) + [observation])
    return stacked_observation, observation_stack


def zeros_like(x):
    """Create a zero state like some state. This handles slightly more complex
    objects such as lists and dictionaries of numpy arrays and torch Tensors.

    Args:
        x (np.ndarray | torch.Tensor | dict | list): State used to define
            structure/state of zero state.
    """
    if isinstance(x, np.ndarray):
        return np.zeros_like(x)
    elif isinstance(x, torch.Tensor):
        return torch.zeros_like(x)
    elif isinstance(x, dict):
        return {k: zeros_like(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [zeros_like(item) for item in x]
    else:
        return 0


def concatenate(xs):
    """Concatenates numpy arrays or dictionaries of numpy arrays.

    Args:
        xs (list): List of objects to concatenate.
    """

    if len(xs) == 0:
        return np.array([])

    if isinstance(xs[0], dict):
        return {k: np.concatenate([x[k] for x in xs], axis=0) for k in xs[0]}
    else:
        return np.concatenate(xs, axis=0)
