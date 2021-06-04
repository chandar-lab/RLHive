import hanabi_learning_environment.rl_env as rl_env
from hanabi_learning_environment.rl_env import HanabiEnv
from hanabi_learning_environment import pyhanabi
from hive.envs.env_spec import EnvSpec
import numpy as np


class HanabiLearningEnv(HanabiEnv):
    """Hanabi Learning Environemnt from https://github.com/deepmind/hanabi-learning-environment."""

    def __init__(self, env_name, num_players=1, **kwargs):
        """
        Args:
            env_name: Name of the environment (NOTE: make sure it is available at gym.envs.registry.all())
        """
        config = {"players": num_players}
        super().__init__(config)

        self.create_env(env_name, num_players, **kwargs)
        self.env_spec = self.create_env_spec(env_name, num_players, **kwargs)

    def create_env(self, env_name, num_players, **kwargs):
        """Function used to create the environment. Subclasses can override this method
        if they are using a gym style environment that needs special logic.
        """
        self._env = rl_env.make(env_name, num_players)

    def create_env_spec(self, env_name, num_players, **kwargs):
        """Function used to create the specification. Subclasses can override this method
        if they are using a gym style environment that needs special logic.
        """
        obs_dim = self.vectorized_observation_shape()[0]
        act_dim = self.num_moves()

        return EnvSpec(
            env_name=env_name,
            obs_dim=[(obs_dim,) for _ in range(num_players)],
            act_dim=[act_dim for _ in range(num_players)],
        )

    def reset(self):
        self.state = self.game.new_initial_state()

        while self.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            self.state.deal_random_card()

        observation, current_player = self.parse_observation(self._env.reset())
        return observation, current_player

    def step(self, action):
        observations, reward, done, info = self._env.step(action)
        observation, current_player = self.parse_observation(observations)
        # info['legal_moves'] = legal_moves

        return (observation, reward, done, current_player, info)

    def parse_observation(self, observations):
        current_player = observations["current_player"]
        hanabi_obs = observations["player_observations"][current_player]
        legal_moves_as_int = hanabi_obs["legal_moves_as_int"]
        hanabi_obs["vectorized"] = np.array(hanabi_obs["vectorized"])

        return hanabi_obs, current_player
