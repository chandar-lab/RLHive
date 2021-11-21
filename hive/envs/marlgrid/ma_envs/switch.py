import numpy as np
from gym_minigrid.rendering import fill_coords, point_in_rect
from marlgrid.base import MultiGrid
from marlgrid.objects import Floor, Goal, GridAgent

from hive.envs.marlgrid.ma_envs.base import MultiGridEnvHive


class SwitchMultiGrid(MultiGridEnvHive):
    """
    Checkers environment based on sunehag et al. 2017

    "... The map contains apples and lemons. The first player is very sensitive and scores 10 for
    the team for an apple (green square) and −10 for a lemon (orange square).
    The second, less sensitive player scores 1 for the team for an apple and −1 for a lemon.
    There is a wall of lemons between the players and the apples.
    Apples and lemons disappear when collected.
    The environment resets when all apples are eaten or maximum number of steps is reached.
    """

    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)
        for row in range(height - 2):
            if row != (height - 2) // 2:
                self.grid.horz_wall(3, row + 1, width - 6)

        self.put_obj(SimpleFloor(color="blue"), 1, 1)
        self.put_obj(SimpleFloor(color="red"), self.width - 2, self.height - 2)
        self.agent_spawn_kwargs = {}
        self.ghost_mode = False

    def reset(self, **kwargs):
        for agent in self.agents:
            agent.agents = []
            agent.reset(new_episode=True)

        self._gen_grid(self.width, self.height)

        for id, agent in enumerate(self.agents):
            if id == 0:
                top = (0, 0)
            else:
                top = (self.width - 3, 0)
            if agent.spawn_delay == 0:
                self.place_obj(
                    agent,
                    top=top,
                    size=(2, self.height - 1),
                    **self.agent_spawn_kwargs,
                )
                agent.activate()

        self.step_count = 0
        obs = self.gen_obs()
        for ag_idx, _ in enumerate(obs):
            obs[ag_idx] = np.array(obs[ag_idx], dtype=np.uint8)
        return obs

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

        if len(actions) != len(self.agents):
            raise ValueError(
                f"Number of actions is not equal to the number of agents {len(actions)} != {len(self.agents)}"
            )

        step_rewards = np.zeros((len(self.agents)), dtype=np.float)

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
                        if fwd_cell is None or isinstance(fwd_cell, Goal):
                            self.grid.set(*fwd_pos, agent)
                            agent.pos = fwd_pos
                        else:
                            fwd_cell.agents.append(agent)
                            agent.pos = fwd_pos
                            if (
                                isinstance(fwd_cell, Floor)
                                and agent.color == fwd_cell.color
                            ):
                                step_rewards[agent_no] += 5
                                agent.reward(5)
                                agent.done = True

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
                            else:
                                raise ValueError(
                                    "How was agent there in the first place?"
                                )

                        # After moving, the agent shouldn't contain any other agents.
                        agent.agents = []

                        # Rewards can be got iff. fwd_cell has a "get_reward" method
                        if hasattr(fwd_cell, "get_reward"):
                            rwd = fwd_cell.get_reward(agent)

                            # Modify the reward for less sensitive agent
                            if bool(self.reward_decay):
                                rwd *= 1.0 - 0.9 * (self.step_count / self.max_steps)
                            step_rewards[agent_no] += rwd
                            agent.reward(rwd)

                # Pick up an object
                elif action == agent.actions.pickup:
                    if fwd_cell and fwd_cell.can_pickup() and agent.carrying is None:
                        agent.carrying = fwd_cell
                        agent.carrying.cur_pos = np.array([-1, -1])
                        self.grid.set(*fwd_pos, None)

                # Drop an object
                elif action == agent.actions.drop:
                    if not fwd_cell and agent.carrying:
                        self.grid.set(*fwd_pos, agent.carrying)
                        agent.carrying.cur_pos = fwd_pos
                        agent.carrying = None

                # Toggle/activate an object
                elif action == agent.actions.toggle:
                    if fwd_cell:
                        wasted = bool(fwd_cell.toggle(agent, fwd_pos))

                # Done action (not used by default)
                elif action == agent.actions.done:
                    pass

                else:
                    raise ValueError(f"Environment can't handle action {action}.")

                agent.on_step(fwd_cell if agent_moved else None)

        # If any of the agents individually are "done" (hit lava or in some cases a goal)
        # but the env requires respawning, then respawn those agents.
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

        # The episode overall is done if all the agents are done,
        # or if it exceeds the step limit or all the apples are collected.
        done = (self.step_count >= self.max_steps) or all(
            [agent.done for agent in self.agents]
        )

        obs = [
            np.asarray(self.gen_agent_obs(agent), dtype=np.uint8)
            for agent in self.agents
        ]

        # Team reward
        step_rewards = np.array([np.sum(step_rewards) for _ in self.agents])

        return obs, step_rewards, done, {}


# Map of color names to RGB values
COLORS = {
    "red": np.array([255, 0, 0]),
    "blue": np.array([0, 0, 255]),
}


class SimpleFloor(Floor):
    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])
