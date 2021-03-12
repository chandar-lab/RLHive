import gym_minigrid
from gym_minigrid.wrappers import FullyObsWrapper, OneHotPartialObsWrapper, \
     FlatObsWrapper, ImgObsWrapper

from hive.envs.gym_env import GymEnv
from hive.envs.env_spec import EnvSpec


class GymMiniGridEnv(GymEnv):
    """
    Class for loading gym-minigrid environments.
    """

    def __init__(
        self, 
        env_name,
        fully_observable=True,
        one_hot_partial_observable=True,
        flatten_observation=True,
        image_observation=False
    ):
        """
        Args:
            env_name: Name of the environment
            fully_observable: True if fully observable
            one_hot_partial_observable: True for getting the one hot encoded observation
            flatten_observation: True for flattening the image and also the mission
            image_observation: True for only getting the image as an observation and not the mission
        """
        assert not(flatten_observation and image_observation)

        super(GymMiniGridEnv, self).__init__(env_name=env_name)

        if fully_observable:
            self._env = FullyObsWrapper(self._env)
        
        if one_hot_partial_observable:
            self._env = OneHotPartialObsWrapper(self._env)

        if flatten_observation:
            self._env = FlatObsWrapper(self._env)

        if image_observation:
            self._env = ImgObsWrapper(self._env)

        self.env_spec = EnvSpec(env_name=env_name,
                                obs_dim=self._env.observation_space.shape,
                                act_dim=self._env.action_space.n)
        

    def render(self, mode='rgb_array'):
        #TODO 
        pass
