from marlgrid.base import MultiGridEnv, MultiGrid
from hive.envs.ma_envs.base import MultiGridEnvHive
from marlgrid.objects import *
import numpy as np


class CheckersMultiGrid(MultiGridEnvHive):
    mission = "get to the green square"
    metadata = {}

    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)
        apple = Goal(color="green", reward=100)
        orange = Goal(color="red", reward=-100)
        for i in range(int(np.floor(width/2))):
            for j in range(3):
                if (j%2) + 1 + 2*i < width - 1:
                    self.put_obj(orange, (j%2) + 1 + 2*i, j + 1)

                if ((j+1)%2) + 1 + 2*i < width - 1:
                    self.put_obj(apple, ((j+1)%2) + 1 + 2*i, j + 1)

        self.num_remained_goals = 3 * (width - 2)
        self.agent_spawn_kwargs = {}
        self.place_agents(top=(0,4), size=(width, height-4), **self.agent_spawn_kwargs)
        self.ghost_mode = False

    def step(self, actions):
        # Spawn agents if it's time.
        for agent in self.agents:
            if not agent.active and not agent.done and self.step_count >= agent.spawn_delay:
                self.place_obj(agent, **self.agent_spawn_kwargs)
                agent.activate()

        assert len(actions) == len(self.agents)

        step_rewards = np.zeros((len(self.agents, )), dtype=np.float)

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
                        if hasattr(fwd_cell, 'get_reward'):
                            rwd = fwd_cell.get_reward(agent)

                            # Modify the reward for less sensitive agent
                            if agent_no == 0:
                                rwd /= 10
                            if bool(self.reward_decay):
                                rwd *= (1.0 - 0.9 * (self.step_count / self.max_steps))
                            step_rewards[agent_no] += rwd
                            agent.reward(rwd)
                            self.num_remained_goals -= 1

                        if isinstance(fwd_cell, Lava):
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
        done = (self.step_count >= self.max_steps) or all([agent.done for agent in self.agents]) or \
            self.num_remained_goals == 0

        obs = [self.gen_agent_obs(agent) for agent in self.agents]

        return obs, step_rewards, done, {}