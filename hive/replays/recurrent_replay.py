import io
import os
import uuid 
import pickle
import random
import pathlib
import datetime
import collections

import numpy as np

from hive.replays.replay_buffer import BaseReplayBuffer
from hive.utils.utils import create_folder, seeder

class RecurrentReplayBuffer(BaseReplayBuffer):
    """An efficient version of a recurrent replay buffer that only stores each episode
    once.
    """
    # TODO: Do we need self._specs with {observation/action/reward}_{shape/dtype}
    def __init__(
        self,
        directory,
        capacity: int = 10000,
        ongoing: bool = False,
        # stack_size: int = 1,
        length: int = 1,
        minlen: int = 1,
        maxlen: int = 0,
        # gamma: float = 0.99,
        observation_shape=(),
        observation_dtype=np.uint8,
        action_shape=(),
        action_dtype=np.int8,
        reward_shape=(),
        reward_dtype=np.float32,
        extra_storage_types=None,
        prioritize_ends=False,
        num_players_sharing_buffer: int = None,
    ):
        """Constructor for RecurrentReplayBuffer.

        Args:
            capacity (int): Total number of observations that can be stored in the
                buffer. Note, this is not the same as the number of transitions that
                can be stored in the buffer.
            stack_size (int): The number of frames to stack to create an observation.
            n_step (int): Horizon used to compute n-step return reward
            minlen (int): Minimum lenth of an episode to be stored in the buffer.
            maxlen (int): Maximum length of an episode to be stored in the buffer.
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
        self._directory = pathlib.Path(directory).expanduser()
        self._directory.mkdir(parents=True, exist_ok=True)

        self._length = length
        self._capacity = capacity
        self._ongoing = ongoing
        self._minlen = minlen
        self._maxlen = maxlen
        self._prioritize_ends = prioritize_ends
        self._random = np.random.RandomState()

        self._specs = {
            "observation": (observation_dtype, observation_shape),
            "done": (np.uint8, ()),
            "action": (action_dtype, action_shape),
            "reward": (reward_dtype, reward_shape),
        }
        # if extra_storage_types is not None:
        #     self._specs.update(extra_storage_types)

        # self._storage = self._create_storage(capacity, self._specs)
        # self._stack_size = stack_size
        
        # FIXME: Write the function load_episodes
        # Should I make these functions as part of the class? Or should I make them as a separate function like the DV2 repo?
        self._complete_eps = self._load_episodes(self._directory, capacity, minlen)
        # FIXME: Assuming a single episode at the moment
        self._ongoing_eps = collections.defaultdict(list) 
            # lambda: collections.defaultdict(list))
        self._total_episodes, self._total_steps = self._count_episodes(self._directory)
        self._loaded_episodes = len(self._complete_eps)
        self._loaded_steps = sum(self._eplen(x) for x in self._complete_eps.values())
        
        # self._episode_start = True
        # self._cursor = 0
        # self._num_added = 0
        self._num_players_sharing_buffer = num_players_sharing_buffer
        # if num_players_sharing_buffer is not None:
        #     self._episode_storage = [[] for _ in range(num_players_sharing_buffer)]

    def size(self):
        """Returns the number of transitions stored in the buffer."""
        # TODO: do we need only stored ones of what is even outside the buffer?
        return max(
            min(self._num_added, self._capacity) - self._stack_size - self._n_step + 1,
            0,
        )

    def _add_transition(self, **transition):
        """Internal method to add a transition to the buffer."""
        episode = self._ongoing_eps

        for key, value in transition.items():
            episode[key].append(value)
            # TODO: Does transition return "is_last" in hive too? Replace this will done flag. Check if done is the right 
            # thing to do and is done True even when the episode is not complete.
        if transition['done']:
            self._add_episode(episode)
            episode.clear()

    def _eplen(self, episode):
        return len(episode['action']) - 1

    # TODO: Convert should be moved to a common utils file
    def _convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            return value.astype(np.float32)
        elif np.issubdtype(value.dtype, np.signedinteger):
            return value.astype(np.int32)
        elif np.issubdtype(value.dtype, np.uint8):
            return value.astype(np.uint8)
        return value
    
    def _add_episode(self, episode):
        length = self._eplen(episode)

        # print ('here ', length)
        if length < self._minlen:
            print(f'Skipping short episode of length {length}.')
            return
        self._total_steps += length
        self._loaded_steps += length
        self._total_episodes += 1
        self._loaded_episodes += 1
        episode = {key: self._convert(value) for key, value in episode.items()}
        filename = self._save_episode(self._directory, episode)
        self._complete_eps[str(filename)] = episode
        self._enforce_limit()
        
    def _enforce_limit(self):
        if not self._capacity:
          return
        while self._loaded_episodes > 1 and self._loaded_steps > self._capacity:
            # Relying on Python preserving the insertion order of dicts.
            oldest, episode = next(iter(self._complete_eps.items()))
            self._loaded_steps -= self._eplen(episode)
            self._loaded_episodes -= 1
            del self._complete_eps[oldest]
        
    def _save_episode(self, directory, episode):
        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        identifier = str(uuid.uuid4().hex)
        length = self._eplen(episode)
        filename = directory / f'{timestamp}-{identifier}-{length}.npz'
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with filename.open('wb') as f2:
                f2.write(f1.read())
        
        return filename

    def add(self, observation, action, reward, done, **kwargs):
        """Adds a transition to the buffer.
        The required components of a transition are given as positional arguments. The
        user can pass additional components to store in the buffer as kwargs as long as
        they were defined in the specification in the constructor.
        """
        # TODO: This will work as the add_step function in DV2.
        # Will need to add a function to add the episode at the end and save the episode
        # DV2 saves all the episodes to load them again in the future (if required).
        # Question: Does the replay buffer samples episodes using generator functions in Hive?

        # TODO: Might not need this episode_start thing anymore as episode is not added to storage till its complete.
        # if self._episode_start:
        #     self._pad_buffer(self._stack_size - 1)
        #     self._episode_start = False
        transition = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "done": done,
        }

        transition.update(kwargs)
        for key in self._specs:
            obj_type = (
                transition[key].dtype
                if hasattr(transition[key], "dtype")
                else type(transition[key])
            )
            if not np.can_cast(obj_type, self._specs[key][0], casting="same_kind"):
                raise ValueError(
                    f"Key {key} has wrong dtype. Expected {self._specs[key][0]},"
                    f"received {type(transition[key])}."
                )

        # TODO: _num_players_sharing_buffer: What does this mean?
        if self._num_players_sharing_buffer is None:
            self._add_transition(**transition)
        else:
            # FIXME: How to handle this stuff?
            self._episode_storage[kwargs["agent_id"]].append(transition)
            if done:
                for transition in self._episode_storage[kwargs["agent_id"]]:
                    self._add_transition(**transition)
                self._episode_storage[kwargs["agent_id"]] = []

        # if done:
        #     self._episode_start = True

    def _sample_sequence(self):
        episodes = list(self._complete_eps.values())
        if self._ongoing:
            episodes += [
                x for x in self._ongoing_eps.values()
                if self._eplen(x) >= self._minlen]

        episode = self._random.choice(episodes)
        total = len(episode['action'])
        length = total
        if self._maxlen:
            length = min(length, self._maxlen)
        
        # Randomize length to avoid all chunks ending at the same time in case the
        # episodes are all of the same length.
        length -= np.random.randint(self._minlen)
        length = max(self._minlen, length)
        upper = total - length + 1
        if self._prioritize_ends:
            upper += self._minlen
        index = min(self._random.randint(upper), total - length)
        sequence = {
            k: self._convert(v[index: index + length])
            for k, v in episode.items() if not k.startswith('log_')}
        sequence['is_first'] = np.zeros(len(sequence['action']), np.bool)
        sequence['is_first'][0] = True
        if self._maxlen:
            assert self._minlen <= len(sequence['action']) <= self._maxlen
        
        return sequence

    def _generate_chunks(self, length):
        sequence = self._sample_sequence()
        while True:
            chunk = collections.defaultdict(list)
            added = 0
            while added < length:
                needed = length - added
                adding = {k: v[:needed] for k, v in sequence.items()}
                sequence = {k: v[needed:] for k, v in sequence.items()}
                for key, value in adding.items():
                    chunk[key].append(value)
                added += len(adding['action'])
                if len(sequence['action']) < 1:
                    sequence = self._sample_sequence()
            chunk = {k: np.concatenate(v) for k, v in chunk.items()}

            return chunk

    def sample(self, batch_size, length):
        """Sample transitions from the buffer. For a given transition, if it's
        done is True, the next_observation value should not be taken to have any
        meaning.

        Args:
            batch_size (int): Number of transitions to sample.
        """
        # TODO: not the most optimized implementation
        batch = [self._generate_chunks(length) for _ in range(batch_size)]
        # print (batch[0])
        batch = {k: np.array([dic[k] for dic in batch], dtype=batch[0][k].dtype) for k in batch[0] if k != 'info'}
        return batch

    def _load_episodes(self, directory, capacity=None, minlen=1):
        # The returned directory from filenames to episodes is guaranteed to be in
        # temporally sorted order.
        filenames = sorted(directory.glob('*.npz'))
        if capacity:
            num_steps = 0
            num_episodes = 0
            for filename in reversed(filenames):
                length = int(str(filename).split('-')[-1][:-4])
                num_steps += length
                num_episodes += 1
                if num_steps >= capacity:
                    break
            filenames = filenames[-num_episodes:]
        episodes = {}
        for filename in filenames:
            try:
                with filename.open('rb') as f:
                    episode = np.load(f, allow_pickle=True)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f'Could not load episode {str(filename)}: {e}')
                continue
            episodes[str(filename)] = episode
        return episodes

    def _count_episodes(self, directory):
        filenames = list(directory.glob('*.npz'))
        num_episodes = len(filenames)
        num_steps = sum(int(str(n).split('-')[-1][:-4]) - 1 for n in filenames)
        return num_episodes, num_steps
    
    def load(self):
        pass

    def save(self):
        pass

def str_to_dtype(dtype):
    if isinstance(dtype, type) or isinstance(dtype, np.dtype):
        return dtype
    elif dtype.startswith("np.") or dtype.startswith("numpy."):
        return np.sctypeDict[dtype.split(".")[1]]
    else:
        type_dict = {
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
        }
        return type_dict[dtype]
