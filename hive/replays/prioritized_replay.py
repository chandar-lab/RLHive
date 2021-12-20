import os
from typing import Dict, Tuple

import numpy as np

from hive.replays.circular_replay import CircularReplayBuffer
from hive.utils.torch_utils import numpify


class PrioritizedReplayBuffer(CircularReplayBuffer):
    """Implements a replay with prioritized sampling. See
    https://arxiv.org/abs/1511.05952
    """

    def __init__(
        self,
        capacity: int,
        beta: float = 0.5,
        stack_size: int = 1,
        n_step: int = 1,
        gamma: float = 0.9,
        observation_shape: Tuple = (),
        observation_dtype: type = np.uint8,
        action_shape: Tuple = (),
        action_dtype: type = np.int8,
        reward_shape: Tuple = (),
        reward_dtype: type = np.float32,
        extra_storage_types: Dict = None,
        num_players_sharing_buffer=None,
    ):
        """
        Args:
            capacity (int): Total number of observations that can be stored in the
                buffer. Note, this is not the same as the number of transitions that
                can be stored in the buffer.
            beta (float): Parameter controlling level of prioritization.
            stack_size (int): The number of frames to stack to create an observation.
            n_step (int): Horizon used to compute n-step return reward
            gamma (float): Discounting factor used to compute n-step return reward
            observation_shape: Shape of observations that will be stored in the buffer.
            observation_dtype: Type of observations that will be stored in the buffer.
                This can either be the type itself or string representation of the
                type. The type can be either a native python type or a numpy type. If
                a numpy type, a string of the form np.uint8 or numpy.uint8 is
                acceptable.
            action_shape: Shape of actions that will be stored in the buffer.
            action_dtype: Type of actions that will be stored in the buffer. Format is
                described in the description of observation_dtype.
            action_shape: Shape of actions that will be stored in the buffer.
            action_dtype: Type of actions that will be stored in the buffer. Format is
                described in the description of observation_dtype.
            reward_shape: Shape of rewards that will be stored in the buffer.
            reward_dtype: Type of rewards that will be stored in the buffer. Format is
                described in the description of observation_dtype.
            extra_storage_types (dict): A dictionary describing extra items to store
                in the buffer. The mapping should be from the name of the item to a
                (type, shape) tuple.
            num_players_sharing_buffer (int): Number of agents that share their
                buffers. It is used for self-play.
        """
        super().__init__(
            capacity=capacity,
            stack_size=stack_size,
            n_step=n_step,
            gamma=gamma,
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            action_shape=action_shape,
            action_dtype=action_dtype,
            reward_shape=reward_shape,
            reward_dtype=reward_dtype,
            extra_storage_types=extra_storage_types,
            num_players_sharing_buffer=num_players_sharing_buffer,
        )
        self._sum_tree = SumTree(self._capacity)
        self._beta = beta

    def set_beta(self, beta):
        self._beta = beta

    def _add_transition(self, priority=None, **transition):
        if priority is None:
            priority = self._sum_tree.max_recorded_priority
        self._sum_tree.set_priority(self._cursor, priority)
        super()._add_transition(**transition)

    def _pad_buffer(self, pad_length):
        for _ in range(pad_length):
            transition = {
                key: np.zeros_like(self._storage[key][0]) for key in self._storage
            }
            transition["priority"] = 0
            self._add_transition(**transition)

    def _sample_indices(self, batch_size):
        indices = self._sum_tree.stratified_sample(batch_size)
        indices = self._filter_transitions(indices)
        while len(indices) < batch_size:
            new_indices = self._sum_tree.sample(batch_size - len(indices))
            new_indices = self._filter_transitions(new_indices)
            indices = np.concatenate([indices, new_indices])
        return indices

    def _filter_transitions(self, indices):
        indices = super()._filter_transitions(indices - (self._stack_size - 1)) + (
            self._stack_size - 1
        )
        if self._num_added < self._capacity:
            indices = indices[indices < self._cursor - self._n_step]
            indices = indices[indices >= self._stack_size - 1]
        else:
            low = (self._cursor - self._n_step) % self._capacity
            high = (self._cursor + self._stack_size - 1) % self._capacity
            if low < high:
                indices = indices[np.logical_or(indices < low, indices > high)]
            else:
                indices = indices[~np.logical_or(indices >= low, indices <= high)]

        return indices

    def sample(self, batch_size):
        batch = super().sample(batch_size)
        indices = batch["indices"]
        priorities = self._sum_tree.get_priorities(indices)
        weights = (1.0 / (priorities + 1e-10)) ** self._beta
        weights /= np.max(weights)
        batch["weights"] = weights
        return batch

    def update_priorities(self, indices, priorities):
        """Update the priorities of the transitions at the specified indices.

        Args:
            indices: Which transitions to update priorities for. Can be numpy array
                or torch tensor.
            priorities: What the priorities should be updated to. Can be numpy array
                or torch tensor.
        """
        indices = numpify(indices)
        priorities = numpify(priorities)
        indices, unique_idxs = np.unique(indices, return_index=True)
        priorities = priorities[unique_idxs]
        self._sum_tree.set_priority(indices, priorities)

    def save(self, dname):
        super().save(dname)
        self._sum_tree.save(dname)

    def load(self, dname):
        super().load(dname)
        self._sum_tree.load(dname)


class SumTree:
    """Data structure used to implement prioritized sampling. It is implemented
    as a tree where the value of each node is the sum of the values of the subtree
    of the node.
    """

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._depth = int(np.ceil(np.log2(capacity))) + 1
        self._tree = np.zeros(2 ** self._depth - 1)
        self._last_level_start = 2 ** (self._depth - 1) - 1
        self._priorities = self._tree[
            self._last_level_start : self._last_level_start + self._capacity
        ]
        self.max_recorded_priority = 1.0

    def set_priority(self, indices, priorities):
        """Sets the priorities for the given indices.

        Args:
            indices (np.ndarray): Which transitions to update priorities for.
            priorities (np.ndarray): What the priorities should be updated to.
        """
        self.max_recorded_priority = max(self.max_recorded_priority, np.max(priorities))
        indices = self._last_level_start + indices
        diffs = priorities - self._tree[indices]
        for _ in range(self._depth):
            np.add.at(self._tree, indices, diffs)
            indices = (indices - 1) // 2

    def sample(self, batch_size):
        """Sample elements from the sum tree with probability proportional to their
        priority.

        Args:
            batch_size (int): The number of elements to sample.
        """
        indices = self.extract(np.random.rand(batch_size))
        return indices

    def stratified_sample(self, batch_size):
        """Performs stratified sampling using the sum tree.

        Args:
            batch_size (int): The number of elements to sample.
        """
        query_values = (np.arange(batch_size) + np.random.rand(batch_size)) / batch_size
        indices = self.extract(query_values)
        return indices

    def extract(self, queries):
        """Get the elements in the sum tree that correspond to the query.
        For each query, the element that is selected is the one with the greatest
        sum of "previous" elements in the tree, but also such that the sum is not
        a greater proportion of the total sum of priorities than the query.

        Args:
            queries (np.ndarray): Queries to extract. Each element should be
                between 0 and 1.
        """
        queries *= self._tree[0]
        indices = np.zeros(queries.shape[0], dtype=np.int64)
        for i in range(self._depth - 1):
            indices = indices * 2 + 1
            left_child_values = self._tree[indices]
            branch_right = (queries > left_child_values).nonzero()
            indices[branch_right] += 1
            queries[branch_right] -= left_child_values[branch_right]
        return indices - self._last_level_start

    def get_priorities(self, indices):
        """Get the priorities of the elements at indicies.

        Args:
            indices (np.ndarray): The indices to query.
        """
        return self._priorities[indices]

    def save(self, dname):
        np.save(os.path.join(dname, "sumtree.npy"), self._tree)

    def load(self, dname):
        self._tree = np.load(os.path.join(dname, "sumtree.npy"))
        self._priorities = self._tree[
            self._last_level_start : self._last_level_start + self._capacity
        ]
