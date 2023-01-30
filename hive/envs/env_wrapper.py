from typing import List

import gym
import numpy as np

from hive.envs.base import BaseEnv
from hive.envs.env_spec import EnvSpec
from hive.utils.registry import Registrable, registry


class EnvWrapper(Registrable):
    """A wrapper for callables that produce environment wrappers.

    These wrapped callables can be partially initialized through configuration
    files or command line arguments.
    """

    @classmethod
    def type_name(cls):
        """
        Returns:
            "env_wrapper"
        """
        return "env_wrapper"


def apply_wrappers(env, env_wrappers):
    for wrapper in env_wrappers:
        env = wrapper(env)
    return env


registry.register_all(
    EnvWrapper,
    {
        "RecordEpisodeStatistics": gym.wrappers.RecordEpisodeStatistics,
        "ClipAction": gym.wrappers.ClipAction,
        "NormalizeObservation": gym.wrappers.NormalizeObservation,
        "TransformObservation": gym.wrappers.TransformObservation,
        "NormalizeReward": gym.wrappers.NormalizeReward,
        "TransformReward": gym.wrappers.TransformReward,
    },
)
