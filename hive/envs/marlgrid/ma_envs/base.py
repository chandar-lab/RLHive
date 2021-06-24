import gym
from marlgrid.base import MultiGridEnv, MultiGrid
from marlgrid.objects import *
import numpy as np


class MultiGridEnvHive(MultiGridEnv):
    def __init__(
        self,
        agents,
        grid_size=None,
        width=None,
        height=None,
        max_steps=100,
        reward_decay=True,
        seed=1337,
        respawn=False,
        ghost_mode=True,
        agent_spawn_kwargs={},
    ):

        super().__init__(
            agents,
            grid_size,
            width,
            height,
            max_steps,
            reward_decay,
            seed,
            respawn,
            ghost_mode,
            agent_spawn_kwargs,
        )

    def reset(self, **kwargs):
        for agent in self.agents:
            agent.agents = []
            agent.reset(new_episode=True)

        self._gen_grid(self.width, self.height)

        for agent in self.agents:
            if agent.spawn_delay == 0:
                agent.activate()

        self.step_count = 0
        obs = self.gen_obs()
        return obs

    def place_agent(self, agent, top=(0, 0), size=None, rand_dir=True, max_tries=100):
        agent.pos = self.place_obj(agent, top=top, size=size, max_tries=max_tries)

        return agent

    def place_agents(self, top=(0, 0), size=None, rand_dir=True, max_tries=100):
        for agent in self.agents:
            self.place_agent(
                agent, top=top, size=size, rand_dir=rand_dir, max_tries=max_tries
            )
            if hasattr(self, "mission"):
                agent.mission = self.mission
