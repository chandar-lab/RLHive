from abc import ABC, abstractmethod


class BaseEnv(ABC):
    """
    Base class for environments, the learning task e.g. an MDP.
    """

    def __init__(self):
        self._env_spec = None

    @abstractmethod
    def reset(self):
        """
        Resets the state of the environment.

        Returns:
            observation: The initial observation of the new episode.
            turn (int): The index of the agent which should take turn.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        """
        Run one time-step of the environment using the input action.

        Args:
            action: An element of environment's action space.

        Returns:
            observation: Indicates the next state that is an element of environment's observation space.
            reward (float): A scalar reward achieved from the transition.
            done (bool): Indicates whether the episode has ended.
            turn: Indicates which agent should take turn.
            info (dict): Additional custom information.
        """

        raise NotImplementedError

    def render(self, mode='rgb_array'):
        """
        Displays a rendered frame from the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def seed(self, seed=None):
        """
        Reseeds the environment.
        """
        raise NotImplementedError

    def save(self, save_dir):
        """
        Saves the environment.
        """
        raise NotImplementedError

    def load(self, load_dir):
        """
        Loads the environment.
        """
        raise NotImplementedError

    def close(self):
        """
        Additional clean up operations
        """
        raise NotImplementedError

    @property
    def env_spec(self):
        return self._env_spec

    @env_spec.setter
    def env_spec(self, env_spec):
        self._env_spec = env_spec
