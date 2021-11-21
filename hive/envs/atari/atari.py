import cv2
import gym
import numpy as np

from typing import Tuple, Dict, Any

from hive.envs.env_spec import EnvSpec
from hive.envs.gym_env import GymEnv


class AtariEnv(GymEnv):
    """
    Class for loading Atari environments.
    """

    def __init__(
        self,
        game: str,
        frame_skip: int = 4,
        screen_size: int = 84,
        sticky_actions_prob: float = 0.25,
        game_mode: int = 0,
        game_difficulty: int = 0,
        full_action_space: bool = True,
    ):
        """
        Args:
            game (str): Name of the Atari game environment
            frame_skip (int): Number of times the agent takes the same action in the environment
            screen_size (int): Size of the resized frames from the environment
            sticky_actions_prob (float): Probability to repeat actions, see Machado et al., 2018
            game_mode (int): Game mode, see Machado et al., 2018
            game_difficulty (int): Game difficulty,see Machado et al., 2018
            full_action_space (bool): Use full action space?, see ALE
        """
        env_name = "ALE/{}-v5".format(game)

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
        if (sticky_actions_prob < 0.0) or (sticky_actions_prob > 1.0):
            raise ValueError(
                "Sticky Actions Probability should be in [0.,1.], got {}".format(
                    screen_size
                )
            )

        self._frame_skip = frame_skip
        self._screen_size = screen_size

        super().__init__(
            env_name,
            **{
                "sticky_actions_prob": sticky_actions_prob,
                "mode": game_mode,
                "difficulty": game_difficulty,
                "full_action_space": full_action_space,
            }
        )

    def create_env(
        self,
        env_name: str,
        sticky_actions_prob: float = 0.25,
        mode: int = 0,
        difficulty: int = 0,
        full_action_space: bool = True,
        **kwargs
    ):
        """
        Notes:
            * We handle frame skipping by ourselves.
            * We only support grayscale observations.
            * Rendering option is off.
        """
        self._env = gym.make(
            env_name,
            obs_type="grayscale",
            frameskip=1,
            mode=mode,
            difficulty=difficulty,
            repeat_action_probability=sticky_actions_prob,
            full_action_space=full_action_space,
            render_mode=None,
        )

    def create_env_spec(self, env_name: str, **kwargs):
        obs_spaces = self._env.observation_space.shape
        # Used for storing and pooling over two consecutive observations
        self.screen_buffer = [
            np.empty((obs_spaces[0], obs_spaces[1]), dtype=np.uint8),
            np.empty((obs_spaces[0], obs_spaces[1]), dtype=np.uint8),
        ]

        act_spaces = [self._env.action_space]
        return EnvSpec(
            env_name=env_name,
            obs_dim=[(1, self._screen_size, self._screen_size)],
            act_dim=[space.n for space in act_spaces],
        )

    def reset(self) -> Tuple[np.ndarray, int]:
        self.screen_buffer[1] = self._env.reset()
        self.screen_buffer[0].fill(0)
        return self._pool_and_resize(), self._turn

    def step(
        self, action: int = None
    ) -> Tuple[np.ndarray, float, bool, int, Dict[str, Any]]:
        """
        Remarks:
            * Execute self.frame_skips steps taking the action in the the environment.
            * This may execute fewer than self.frame_skip steps in the environment, if the done state is reached.
            * Furthermore, in this case the returned observation should be ignored.
        """
        assert action is not None

        accumulated_reward = 0.0
        done = False
        info = {}
        for time_step in range(self._frame_skip):
            observation, reward, done, info = self._env.step(action)
            accumulated_reward += reward

            if done:
                break
            elif time_step >= self._frame_skip - 2:
                t = time_step - (self._frame_skip - 2)
                self.screen_buffer[t] = observation

        observation = self._pool_and_resize()

        return observation, accumulated_reward, done, self._turn, info

    def _pool_and_resize(self) -> np.ndarray:
        """Transforms two frames into a Nature DQN observation.

        Returns:
          transformed_screen (numpy array): pooled, resized screen.
        """
        # Pool if there are enough screens to do so.
        if self._frame_skip > 1:
            np.maximum(
                self.screen_buffer[0], self.screen_buffer[1], out=self.screen_buffer[1]
            )

        transformed_image = cv2.resize(
            self.screen_buffer[1],
            (self._screen_size, self._screen_size),
            interpolation=cv2.INTER_AREA,
        )
        int_image = np.asarray(transformed_image, dtype=np.uint8)
        return np.expand_dims(int_image, axis=0)
