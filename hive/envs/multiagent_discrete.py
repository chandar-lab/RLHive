import gym
from hive.envs import ParallelEnv, GymEnv
from hive.envs.env_spec import EnvSpec
import hive.envs.mabc

class MultiAgentDiscEnv(ParallelEnv, GymEnv):
    """
    MultiAgentDiscEnv is for environments which have discrete observations.
    """
    def __init__(self, env_name, num_players, **kwargs):
        super().__init__(env_name, num_players)

    def create_env(self, env_name, **kwargs):
        super().create_env(env_name, **kwargs)

    def create_env_spec(self, env_name, **kwargs):
        """
        Function used to create the environment specification for discrete
        observations.
        """
        if isinstance(self._env.observation_space, gym.spaces.Tuple):
            obs_spaces = self._env.observation_space.spaces
        else:
            obs_spaces = [self._env.observation_space]
        if isinstance(self._env.action_space, gym.spaces.Tuple):
            act_spaces = self._env.action_space.spaces
        else:
            act_spaces = [self._env.action_space]

        # `num_disc_obs` is the number of discrete observations that each array
        # cell can have. Just a single value is used to represent this value
        # for each cell of the array uniformly (in case of MultiDiscrete).
        # But it can be extended to have a different value for each cell.
        if isinstance(obs_spaces[0], gym.spaces.Discrete):
            num_disc_obs = obs_spaces[0].n
        elif isinstance(obs_spaces[0], gym.spaces.MultiDiscrete):
            num_disc_obs = obs_spaces[0].nvec[0]
        else:
            assert True, "Observation space should be object of type\
            `gym.spaces.Discrete` or `gym.spaces.MultiDiscrete`."

        obs_dim = []
        for space in obs_spaces:
            if space.shape == ():
                obs_dim.append((1,))
            else:
                obs_dim.append(space.shape)
        act_dim = []
        for space in act_spaces:
            if isinstance(space, gym.spaces.Discrete):
                act_dim.append(space.n)
            else:
                assert True, "Action space should be object of type\
                `gym.spaces.Discrete`."
        env_info = {}
        env_info["num_disc_per_obs_dim"] = num_disc_obs

        
        return EnvSpec(
            env_name=env_name,
            obs_dim=obs_dim,
            act_dim=act_dim,
            env_info=env_info,
        )