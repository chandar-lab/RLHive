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
        if grid_size is not None:
            assert width == None and height == None
            width, height = grid_size, grid_size

        self.respawn = respawn

        self.window = None

        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.reward_decay = reward_decay
        self.seed(seed=seed)
        self.agent_spawn_kwargs = agent_spawn_kwargs
        self.ghost_mode = ghost_mode

        self.agents = []
        for agent in agents:
            self.add_agent(agent)

        self.reset()

    def reset(self, **kwargs):
        for agent in self.agents:
            agent.agents = []
            agent.reset(new_episode=True)

        self._gen_grid(self.width, self.height)

        for agent in self.agents:
            if agent.spawn_delay == 0:
                # self.place_obj(agent, **self.agent_spawn_kwargs)
                agent.activate()

        self.step_count = 0
        obs = self.gen_obs()
        return obs

    def gen_agent_obs(self, agent):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """
        grid, vis_mask = self.gen_obs_grid(agent)

        grid_image = grid.render(
            tile_size=agent.view_tile_size, visible_mask=vis_mask, top_agent=agent
        )
        if agent.observation_style == "image":
            return grid_image
        else:
            ret = {"pov": grid_image}
            if agent.observe_rewards:
                ret["reward"] = getattr(agent, "step_reward", 0)
            if agent.observe_position:
                agent_pos = agent.pos if agent.pos is not None else (0, 0)
                ret["position"] = np.array(agent_pos) / np.array(
                    [self.width, self.height], dtype=np.float
                )
            if agent.observe_orientation:
                agent_dir = agent.dir if agent.dir is not None else 0
                ret["orientation"] = agent_dir
            return ret

    def gen_obs(self):
        return [self.gen_agent_obs(agent) for agent in self.agents]

    def step(self, actions):
        # Spawn agents if it's time.
        for agent in self.agents:
            if (
                not agent.active
                and not agent.done
                and self.step_count >= agent.spawn_delay
            ):
                self.place_obj(agent, **self.agent_spawn_kwargs)
                agent.activate()

        assert len(actions) == len(self.agents)

        step_rewards = np.zeros(
            (
                len(
                    self.agents,
                )
            ),
            dtype=np.float,
        )

        self.step_count += 1

        iter_agents = list(enumerate(zip(self.agents, actions)))
        iter_order = np.arange(len(iter_agents))
        self.np_random.shuffle(iter_order)
        for shuffled_ix in iter_order:
            agent_no, (agent, action) = iter_agents[shuffled_ix]
            agent.step_reward = 0

            if agent.active:

                cur_pos = agent.pos[:]
                cur_cell = self.grid.get(*cur_pos)
                fwd_pos = agent.front_pos[:]
                fwd_cell = self.grid.get(*fwd_pos)
                agent_moved = False

                # Rotate left
                if action == agent.actions.left:
                    agent.dir = (agent.dir - 1) % 4

                # Rotate right
                elif action == agent.actions.right:
                    agent.dir = (agent.dir + 1) % 4

                # Move forward
                elif action == agent.actions.forward:
                    # Under the follow conditions, the agent can move forward.
                    can_move = fwd_cell is None or fwd_cell.can_overlap()
                    if self.ghost_mode is False and isinstance(fwd_cell, GridAgent):
                        can_move = False

                    if can_move:
                        agent_moved = True
                        # Add agent to new cell
                        if fwd_cell is None:
                            self.grid.set(*fwd_pos, agent)
                            agent.pos = fwd_pos
                        else:
                            fwd_cell.agents.append(agent)
                            agent.pos = fwd_pos

                        # Remove agent from old cell
                        if cur_cell == agent:
                            self.grid.set(*cur_pos, None)
                        else:
                            assert cur_cell.can_overlap()
                            cur_cell.agents.remove(agent)

                        # Add agent's agents to old cell
                        for left_behind in agent.agents:
                            cur_obj = self.grid.get(*cur_pos)
                            if cur_obj is None:
                                self.grid.set(*cur_pos, left_behind)
                            elif cur_obj.can_overlap():
                                cur_obj.agents.append(left_behind)
                            else:  # How was "agent" there in teh first place?
                                raise ValueError("?!?!?!")

                        # After moving, the agent shouldn't contain any other agents.
                        agent.agents = []
                        # test_integrity(f"After moving {agent.color} fellow")

                        # Rewards can be got iff. fwd_cell has a "get_reward" method
                        if hasattr(fwd_cell, "get_reward"):
                            rwd = fwd_cell.get_reward(agent)
                            if bool(self.reward_decay):
                                rwd *= 1.0 - 0.9 * (self.step_count / self.max_steps)
                            step_rewards[agent_no] += rwd
                            agent.reward(rwd)

                        if isinstance(fwd_cell, (Lava, Goal)):
                            agent.done = True

                # TODO: verify pickup/drop/toggle logic in an environment that
                #  supports the relevant interactions.
                # Pick up an object
                elif action == agent.actions.pickup:
                    if fwd_cell and fwd_cell.can_pickup():
                        if agent.carrying is None:
                            agent.carrying = fwd_cell
                            agent.carrying.cur_pos = np.array([-1, -1])
                            self.grid.set(*fwd_pos, None)
                    else:
                        pass

                # Drop an object
                elif action == agent.actions.drop:
                    if not fwd_cell and agent.carrying:
                        self.grid.set(*fwd_pos, agent.carrying)
                        agent.carrying.cur_pos = fwd_pos
                        agent.carrying = None
                    else:
                        pass

                # Toggle/activate an object
                elif action == agent.actions.toggle:
                    if fwd_cell:
                        wasted = bool(fwd_cell.toggle(agent, fwd_pos))
                    else:
                        pass

                # Done action (not used by default)
                elif action == agent.actions.done:
                    pass

                else:
                    raise ValueError(f"Environment can't handle action {action}.")

                agent.on_step(fwd_cell if agent_moved else None)

        # If any of the agents individually are "done" (hit lava or in some cases a goal)
        #   but the env requires respawning, then respawn those agents.
        for agent in self.agents:
            if agent.done:
                if self.respawn:
                    resting_place_obj = self.grid.get(*agent.pos)
                    if resting_place_obj == agent:
                        if agent.agents:
                            self.grid.set(*agent.pos, agent.agents[0])
                            agent.agents[0].agents += agent.agents[1:]
                        else:
                            self.grid.set(*agent.pos, None)
                    else:
                        resting_place_obj.agents.remove(agent)
                        resting_place_obj.agents += agent.agents[:]
                        agent.agents = []

                    agent.reset(new_episode=False)
                    self.place_obj(agent, **self.agent_spawn_kwargs)
                    agent.activate()
                else:  # if the agent shouldn't be respawned, then deactivate it.
                    agent.deactivate()

        # The episode overall is done if all the agents are done, or if it exceeds the step limit.
        done = (self.step_count >= self.max_steps) or all(
            [agent.done for agent in self.agents]
        )

        obs = [self.gen_agent_obs(agent) for agent in self.agents]

        return obs, step_rewards, done, {}

    def place_agent(self, agent, top=(0, 0), size=None, rand_dir=True, max_tries=100):
        agent.pos = self.place_obj(agent, top=top, size=size, max_tries=max_tries)
        # if rand_dir:
        #     agent.dir = self._rand_int(0, 4)
        return agent

    def place_agents(self, top=(0, 0), size=None, rand_dir=True, max_tries=100):
        for agent in self.agents:
            self.place_agent(
                agent, top=top, size=size, rand_dir=rand_dir, max_tries=max_tries
            )
            if hasattr(self, "mission"):
                agent.mission = self.mission
