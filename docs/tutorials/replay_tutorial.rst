.. _replays:

Replays
==================

RLHive currently provides 4 types of Replays:

* :py:class:`~hive.replays.circular_replay.CircularReplayBuffer`: An implementation
  of a FIFO circular replay buffer. Stores individual observations and constructs
  transitions on the fly when sampling to save space.
* :py:class:`~hive.replays.circular_replay.SimpleReplayBuffer`: A simplified version
  of a FIFO circular replay buffer that stores individual transitions directly.
* :py:class:`~hive.replays.prioritized_replay.PrioritizedReplayBuffer`: A subclass
  of :py:class:`~hive.replays.circular_replay.CircularReplayBuffer` that adds 
  prioritized sampling.
* :py:class:`~hive.replays.legal_moves_replay.LegalMovesReplayBuffer`: A subclass
  of :py:class:`~hive.replays.prioritized_replay.PrioritizedReplayBuffer` that 
  stores/handles legal moves.

The main replay buffer classes that you will likely use/extend are
:py:class:`~hive.replays.circular_replay.CircularReplayBuffer` and
:py:class:`~hive.replays.prioritized_replay.PrioritizedReplayBuffer`.
By default, these classes expect the arguments ``"observation"``, ``"action"``,
``"reward"``, and ``"done"`` when adding to the buffer. You can also provide alternative
shapes/dtypes for these keys, and the buffer will try to automatically cast the objects
you add to the buffer. 

Along with these default keys, you can also store extra keys in the buffer. When
creating the buffer, provide a dictionary with key-value pairs ``key: (type, shape)``.
When adding, you can directly provide this key as an argument to the 
:py:meth:`~hive.replays.circular_replay.CircularReplayBuffer.add` method, and it will
automatically be added to the batch dictionary that you sample.
