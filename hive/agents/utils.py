import numpy as np
import torch


def get_stacked_state(observation, observation_stack, stack_size):
    """Create a stacked state for the agent. The previous observations recorded
    by this agent are stacked with the current observation. If not enough
    observations have been recorded, zero arrays are appended.

    Args:
        observation: Current observation.
        observation_stack: Observations from previous timesteps.
        stack_size: Number of observations to stack.
    """

    if stack_size == 1:
        return observation, observation_stack
    while len(observation_stack) < stack_size - 1:
        observation_stack.append(zeros_like(observation))

    stacked_observation = concatenate(list(observation_stack) + [observation])
    return stacked_observation, observation_stack


def roll_state(state, stack):
    if isinstance(state, np.ndarray):
        # state = D x ...
        # stack = S x D ...
        stack = np.roll(stack, -1, axis=0)
        stack[-1] = state
        return stack
    elif isinstance(state, dict):
        return {k: roll_state(v, stack[k]) for k, v in state.items()}
    elif isinstance(state, list):
        return [zeros_like(*item) for item in zip(state, stack)]
    elif isinstance(state, tuple):
        return tuple(zeros_like(*item) for item in zip(state, stack))
    else:
        return 0


def get_multiple_zeros_like(x, tile=1):
    """Create a zero state like some state, with the 0th dimension tiled.
    This handles slightly more complex objects such as lists, dictionaries, and
    tuples of numpy arrays.

    Args:
        x (np.ndarray | torch.Tensor | dict | list): State used to define
            structure/state of zero state.
    """
    if isinstance(x, np.ndarray):
        x = np.concatenate([x] * tile, axis=0)
        return np.zeros_like(x)
    elif isinstance(x, dict):
        return {k: zeros_like(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [zeros_like(item) for item in x]
    elif isinstance(x, tuple):
        return tuple(zeros_like(item) for item in x)
    else:
        return 0


def zeros_like(x):
    """Create a zero state like some state. This handles slightly more complex
    objects such as lists, dictionaries, and tuples of numpy arrays.

    Args:
        x (np.ndarray | torch.Tensor | dict | list): State used to define
            structure/state of zero state.
    """
    if isinstance(x, np.ndarray):
        return np.zeros_like(x)
    elif isinstance(x, dict):
        return {k: zeros_like(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [zeros_like(item) for item in x]
    elif isinstance(x, tuple):
        return tuple(zeros_like(item) for item in x)
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
