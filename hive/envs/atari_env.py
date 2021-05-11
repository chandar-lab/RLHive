import numpy as np
import cv2

from hive.envs.gym_env import GymEnv
from hive.envs.env_spec import EnvSpec


class AtariEnv(GymEnv):
    """
        Class for loading Atari environments.
    """

    def __init__(
            self,
            env_name,
            sticky_actions=True,
            frame_skip=4,
            screen_size=84,
    ):
        env_version = 'v0' if sticky_actions else 'v4'
        full_env_name = '{}NoFrameskip-{}'.format(env_name, env_version)
        super().__init__(full_env_name)

        if frame_skip <= 0:
            raise ValueError('Frame skip should be strictly positive, got {}'.
                             format(frame_skip))
        if screen_size <= 0:
            raise ValueError('Target screen size should be strictly positive, got {}'.
                             format(screen_size))

        self.frame_skip = frame_skip
        self.screen_size = screen_size

        # Used for storing and pooling over two consecutive observations
        obs_dims = self.env_spec.obs_dim
        self.screen_buffer = [
            np.empty((1, obs_dims[0][0], obs_dims[0][1]), dtype=np.uint8),
            np.empty((1, obs_dims[0][0], obs_dims[0][1]), dtype=np.uint8)
        ]

        # Changing the observation space to the screen size
        self.env_spec = EnvSpec(
            self.env_spec.env_name,
            [(1, self.screen_size, self.screen_size)],
            self.env_spec.act_dim,
            self.env_spec.env_info
        )

    def reset(self):
        self._env.reset()
        self._get_observation_screen(self.screen_buffer[0])
        self.screen_buffer[1].fill(0)
        return self._pool_and_resize(), self._turn

    def step(self, action=None):
        """
        Remarks:
            * Execute self.frame_skips steps taking the action in the the environment.
            * This may execute fewer than self.frame_skip steps in the environment, if the done state is reached.
            * Furthermore, in this case the returned observation should be ignored.
        """
        assert action is not None

        accumulated_reward = 0.
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
          output (numpy array): screen buffer to hold the returned observation.

        Returns:
          observation (numpy array): the current observation in grayscale.
        """
        self._env.ale.getScreenGrayscale(output)
        return output

    def _pool_and_resize(self):
        """Transforms two frames into a Nature DQN observation.

        For efficiency, the transformation is done in-place in self.screen_buffer.

        Returns:
          transformed_screen (numpy array): pooled, resized screen.
        """
        # Pool if there are enough screens to do so.
        if self.frame_skip > 1:
            np.maximum(self.screen_buffer[0], self.screen_buffer[1],
                       out=self.screen_buffer[0])

        transformed_image = cv2.resize(self.screen_buffer[0],
                                       (self.screen_size, self.screen_size),
                                       interpolation=cv2.INTER_AREA)
        int_image = np.asarray(transformed_image, dtype=np.uint8)
        return np.expand_dims(int_image, axis=2)
