from abc import ABC, abstractmethod


class BaseEnv(ABC):
    """
    Base class for environments, the learning task e.g. an MDP.
    """

    def __init__(self, env_spec, num_players):
        self._env_spec = env_spec
        self._num_players = num_players
        self._turn = 0

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

    def render(self, mode="rgb_array"):
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


class ParallelEnv(BaseEnv):
    def __init__(self, env_name, num_players):
        super().__init__(env_name, num_players)
        self._actions = []
        self._obs = None
        self._info = None
        self._done = False

    def reset(self):
        self._obs, _ = super().reset()
        return self._obs[0], 0

    def step(self, action):
        self._actions.append(action)
        if len(self._actions) == self._num_players:
            observation, reward, done, _, info = super().step(self._actions)
            self._actions = []
            self._turn = 0
            self._obs = observation
            self._info = info
            self._done = done
        else:
            self._turn = (self._turn + 1) % self._num_players
            reward = 0
        return (
            self._obs[self._turn],
            reward,
            self._done and self._turn == 0,
            self._turn,
            self._info,
        )
