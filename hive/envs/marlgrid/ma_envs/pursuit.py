import numpy as np
from marlgrid.base import MultiGrid
from marlgrid.objects import Goal, GridAgent, Lava, Wall

from hive.envs.marlgrid.ma_envs.base import MultiGridEnvHive


class PursuitMultiGrid(MultiGridEnvHive):
    """
    Pursuit–Evasion environment based on Gupta et al. 2017

    "The pursuit-evasion domain consists of two sets of agents: evaders and pursuers.
    The evaders are trying to avoid pursuers, while the pursuers are
    trying to catch the evaders. The pursuers receive a reward of 5.0 when
    they surround an evader or corner the agent"
    """

    metadata = {}

    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)
        self.ghost_mode = False

    def reset(self, **kwargs):
        obs = super().reset()
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

        num_learning_agents = len(actions)
        num_rand_agents = self.num_agents - len(actions)

        step_rewards = np.zeros((num_learning_agents), dtype=np.float)

        self.step_count += 1
        for i in range(num_learning_agents, self.num_agents):
            actions.append(self.action_space[i].sample())
        iter_agents = list(enumerate(zip(self.agents, actions)))
        iter_order = np.arange(len(iter_agents))
        for shuffled_ix in iter_order:
            agent_no, (agent, action) = iter_agents[shuffled_ix]
            agent.step_reward = 0

            if agent.active:

                cur_pos = agent.pos[:]
                cur_cell = self.grid.get(*cur_pos)
                fwd_pos = agent.front_pos[:]
                fwd_cell = self.grid.get(*fwd_pos)
                agent_moved = False
                bot_pos = agent.pos + np.array([0, -1])
                bot_cell = self.grid.get(*bot_pos)
                abov_pos = agent.pos + np.array([0, +1])
                abov_cell = self.grid.get(*abov_pos)
                left_pos = agent.pos + np.array([-1, 0])
                left_cell = self.grid.get(*left_pos)
                right_pos = agent.pos + np.array([+1, 0])
                right_cell = self.grid.get(*right_pos)

                w = 0
                a = 0
                surrounding_cells = [bot_cell, abov_cell, left_cell, right_cell]
                if agent_no == len(self.agents) - num_rand_agents:
                    for cell in surrounding_cells:
                        if isinstance(cell, GridAgent):
                            a += 1
                        if isinstance(cell, Wall):
                            w += 1

                    if a == len(self.agents) - num_rand_agents or (w == 2 and a == 2):
                        step_rewards[:] += np.array(
                            [5] * (len(self.agents) - num_rand_agents)
                        )
                        for agent in self.agents:
                            agent.done = True

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
                            else:
                                raise ValueError(
                                    "How was agent there in teh first place?"
                                )

                        # After moving, the agent shouldn't contain any other agents.
                        agent.agents = []

                        # Rewards can be got iff. fwd_cell has a "get_reward" method
                        if hasattr(fwd_cell, "get_reward"):
                            rwd = fwd_cell.get_reward(agent)
                            if bool(self.reward_decay):
                                rwd *= 1.0 - 0.9 * (self.step_count / self.max_steps)
                            step_rewards[agent_no] += rwd
                            agent.reward(rwd)

                        if isinstance(fwd_cell, (Lava, Goal)):
                            agent.done = True

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
        # or if it exceeds the step limit.
        done = (self.step_count >= self.max_steps) or all(
            [agent.done for agent in self.agents[:num_learning_agents]]
        )

        obs = [
            np.asarray(self.gen_agent_obs(agent), dtype=np.uint8)
            for agent in self.agents[:num_learning_agents]
        ]

        return obs, step_rewards, done, {}
