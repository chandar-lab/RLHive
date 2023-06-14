import numpy as np
import optree


def roll_state(state, stack):
    def _roll_state(state, stack):
        stack = np.roll(stack, -1, axis=0)
        stack[-1] = state
        return stack

    return optree.tree_map(_roll_state, state, stack)


def zeros_like(x):
    """Create a zero state like some state. This handles slightly more complex
    objects such as lists, dictionaries, and tuples of numpy arrays.

    Args:
        x (np.ndarray | dict | list): State used to define
            structure/state of zero state.
    """
    return optree.tree_map(lambda x: np.zeros_like(x), x)


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
