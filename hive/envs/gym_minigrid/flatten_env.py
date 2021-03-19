import gym_minigrid
from gym_minigrid.wrappers import FullyObsWrapper, FlatObsWrapper

from hive.envs.gym_env import GymEnv
from hive.envs.env_spec import EnvSpec
from hive.envs.gym_minigrid.wrappers import FlatObsNoMissionWrapper


class GymMiniGridFlattenEnv(GymEnv):
    """
    Class for loading gym-minigrid environments in which the observation is a one dimension vector.
    """

    def __init__(
            self,
            env_name,
            fully_observable=True,
            encode_mission=False
    ):
        """
        Args:
            env_name: Name of the environment
            fully_observable: True if fully observable
            encode_mission: True if mission should be encoded in the flatten vector
        """

        super(GymMiniGridFlattenEnv, self).__init__(env_name=env_name)

        if fully_observable:
            self._env = FullyObsWrapper(self._env)

        if encode_mission:
            self._env = FlatObsWrapper(self._env)
        else:
            self._env = FlatObsNoMissionWrapper(self._env)

        self.env_spec = EnvSpec(env_name=env_name,
                                obs_dim=self._env.observation_space.shape,
                                act_dim=self._env.action_space.n)

    def render(self, mode='rgb_array'):
        # TODO
        pass
