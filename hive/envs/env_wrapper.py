from hive.utils.registry import Registrable
import gymnasium as gym

class GymWrapper(Registrable, gym.core.Wrapper):
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
