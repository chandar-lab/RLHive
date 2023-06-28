from typing import Generic, Sequence, TypeVar, Union

import gymnasium as gym

OSpace = TypeVar("OSpace", bound=gym.Space)
ASpace = TypeVar("ASpace", bound=gym.Space)


class EnvSpec(Generic[OSpace, ASpace]):
    """Object used to store information about environment configuration.
    Every environment should create an EnvSpec object.
    """

    def __init__(
        self,
        env_name,
        observation_space: Union[OSpace, Sequence[OSpace]],
        action_space: Union[ASpace, Sequence[ASpace]],
        env_info=None,
    ):
        """
        Args:
            env_name: Name of the environment
            observation_space: Spaces of observations from environment. This should be
                a single instance or list of gym.Space, depending on if the environment
                is multiagent.
            action_space: Spaces of actions expected by environment. This should be
                a single instance or list of gym.Space, depending on if the environment
                is multiagent.
            env_info: Any other info relevant to this environment. This can
                include items such as random seeds or parameters used to create
                the environment
        """
        self._env_name = env_name
        self._observation_space = observation_space
        self._action_space = action_space
        self._env_info = {} if env_info is None else env_info

    @property
    def env_name(self):
        return self._env_name

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def env_info(self):
        return self._env_info
