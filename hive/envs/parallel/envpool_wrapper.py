import envpool
import gymnasium as gym
import numpy as np

from hive.envs import BaseEnv
from hive.envs.env_spec import EnvSpec
from hive.utils.utils import seeder


class EnvPool(BaseEnv):
    """
    Class for loading gym environments.
    """

    def __init__(
        self,
        env_name,
        env_type="gymnasium",
        num_envs=1,
        batch_size=1,
        num_players=1,
        **kwargs
    ):
        """
        Args:
            env_name (str): Name of the environment (NOTE: make sure it is available
                in gym.envs.registry.all())
            num_players (int): Number of players for the environment.
            render_mode (str): One of None, "human", "rgb_array", "ansi", or
                "rgb_array_list". See gym documentation for details.
            kwargs: Any arguments you want to pass to :py:meth:`create_env` or
                :py:meth:`create_env_spec` can be passed as keyword arguments to this
                constructor.
        """
        kwargs["seed"] = kwargs.get("seed", seeder.get_new_seed("env"))
        self.create_env(
            env_name,
            env_type=env_type,
            num_envs=num_envs,
            batch_size=batch_size,
            max_num_players=num_players,
            **kwargs,
        )
        super().__init__(
            self.create_env_spec(
                env_name,
                num_envs=num_envs,
                batch_size=batch_size,
                max_num_players=num_players,
                **kwargs,
            ),
            num_players,
        )
        self._output = None

    def create_env(self, env_name, **kwargs):
        """Function used to create the environment. Subclasses can override this method
        if they are using a gym style environment that needs special logic.

        Args:
            env_name (str): Name of the environment
        """
        self._env = envpool.make(env_name, **kwargs)

    def create_env_spec(self, env_name, **kwargs):
        """Function used to create the specification. Subclasses can override this method
        if they are using a gym style environment that needs special logic.

        Args:
            env_name (str): Name of the environment
        """
        if isinstance(self._env.observation_space, gym.spaces.Tuple):
            observation_spaces = self._env.observation_space.spaces
        else:
            observation_spaces = [self._env.observation_space]
        if isinstance(self._env.action_space, gym.spaces.Tuple):
            action_spaces = self._env.action_space.spaces
        else:
            action_spaces = [self._env.action_space]

        return EnvSpec(
            env_name=env_name,
            observation_space=observation_spaces,
            action_space=action_spaces,
        )

    def reset(self):
        self._env.recv()
        self._env.async_reset()

    def get_next_observations(self):
        if self._output is not None:
            obs, info = self._output
            num_envs = obs.shape[0]
            rew = np.zeros(num_envs)
            term = np.zeros(num_envs, dtype=bool)
            trunc = np.zeros(num_envs, dtype=bool)
            self._output = None
        else:
            obs, rew, term, trunc, info = self._env.recv()

        return info["env_id"], (obs, rew, term, trunc, 0, info)

    def send(self, action, env_id):
        self._env.send(action, env_id)

    def step(self, action):
        observation, reward, terminated, truncated, info = self._env.step(action)
        self._turn = (self._turn + 1) % self._num_players
        return observation, reward, terminated, truncated, self._turn, info

    def render(self):
        return self._env.render()

    def seed(self, seed=None):
        self._seed = seed

    def close(self):
        self._env.close()
