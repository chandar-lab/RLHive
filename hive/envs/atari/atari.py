import ale_py
import cv2
import numpy as np

from hive.envs.env_spec import EnvSpec
from hive.envs.gym_env import GymEnv


class AtariEnv(GymEnv):
    """
    Class for loading Atari environments.

    Adapted from the Dopamine's Atari preprocessing code:
    https://github.com/google/dopamine/blob/6fbb58ad9bc1340f42897e8a551f85a01fb142ce/dopamine/discrete_domains/atari_lib.py
    Licensed under Apache 2.0, https://github.com/google/dopamine/blob/master/LICENSE
    """

    def __init__(
        self,
        env_name,
        frame_skip=4,
        screen_size=84,
        sticky_actions=True,
    ):
        """
        Args:
            env_name (str): Name of the environment
            sticky_actions (bool): Whether to use sticky_actions as per Machado et al.
            frame_skip (int): Number of times the agent takes the same action in the environment
            screen_size (int): Size of the resized frames from the environment
        """
        env_version = "v0" if sticky_actions else "v4"
        full_env_name = "{}NoFrameskip-{}".format(env_name, env_version)

        if frame_skip <= 0:
            raise ValueError(
                "Frame skip should be strictly positive, got {}".format(frame_skip)
            )
        if screen_size <= 0:
            raise ValueError(
                "Target screen size should be strictly positive, got {}".format(
                    screen_size
                )
            )

        self.frame_skip = frame_skip
        self.screen_size = screen_size

        super().__init__(full_env_name)

    def create_env_spec(self, env_name, **kwargs):
        obs_spaces = self._env.observation_space.shape
        # Used for storing and pooling over two consecutive observations
        self.screen_buffer = [
            np.empty((obs_spaces[0], obs_spaces[1]), dtype=np.uint8),
            np.empty((obs_spaces[0], obs_spaces[1]), dtype=np.uint8),
        ]

        act_spaces = [self._env.action_space]
        return EnvSpec(
            env_name=env_name,
            obs_dim=[(1, self.screen_size, self.screen_size)],
            act_dim=[space.n for space in act_spaces],
        )

    def reset(self):
        self._env.reset()
        self._get_observation_screen(self.screen_buffer[1])
        self.screen_buffer[0].fill(0)
        return self._pool_and_resize(), self._turn

    def step(self, action=None):
        """
        Remarks:
            * Executes the action for :attr:`self.frame_skips` steps in the the
              environment.
            * This may execute fewer than self.frame_skip steps in the environment, if
              the done state is reached.
            * In this case the returned observation should be ignored.
        """
        assert action is not None

        accumulated_reward = 0.0
        done = False
        info = {}

        for time_step in range(self.frame_skip):
            _, reward, done, info = self._env.step(action)
            accumulated_reward += reward

            if done:
                break
            elif time_step >= self.frame_skip - 2:
                t = time_step - (self.frame_skip - 2)
                self._get_observation_screen(self.screen_buffer[t])

        observation = self._pool_and_resize()

        return observation, accumulated_reward, done, self._turn, info

    def _get_observation_screen(self, output):
        """Get the screen input of the current observation given empty numpy array in grayscale.

        Args:
          output (np.ndarray): screen buffer to hold the returned observation.

        Returns:
          observation (np.ndarray): the current observation in grayscale.
        """
        self._env.ale.getScreenGrayscale(output)
        return output

    def _pool_and_resize(self):
        """Transforms two frames into a Nature DQN observation.

        Returns:
          transformed_screen (np.ndarray): pooled, resized screen.
        """
        # Pool if there are enough screens to do so.
        if self.frame_skip > 1:
            np.maximum(
                self.screen_buffer[0], self.screen_buffer[1], out=self.screen_buffer[1]
            )

        transformed_image = cv2.resize(
            self.screen_buffer[1],
            (self.screen_size, self.screen_size),
            interpolation=cv2.INTER_AREA,
        )
        int_image = np.asarray(transformed_image, dtype=np.uint8)
        return np.expand_dims(int_image, axis=0)
