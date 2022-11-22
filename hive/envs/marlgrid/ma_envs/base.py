import gym
import numpy as np
from marlgrid.base import MultiGrid, MultiGridEnv, rotate_grid
from marlgrid.rendering import SimpleImageViewer

TILE_PIXELS = 32


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
        full_obs=False,
        agent_spawn_kwargs={},
    ):
        self._full_obs = full_obs
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

    def gen_obs_grid(self, agent):
        # If the agent is inactive, return an empty grid and a visibility mask that hides everything.
        if not agent.active:
            # below, not sure orientation is correct but as of 6/27/2020 that doesn't matter because
            # agent views are usually square and this grid won't be used for anything.
            grid = MultiGrid(
                (agent.view_size, agent.view_size), orientation=agent.dir + 1
            )
            vis_mask = np.zeros((agent.view_size, agent.view_size), dtype=np.bool)
            return grid, vis_mask

        if self._full_obs:
            topX, topY, botX, botY = 0, 0, self.width, self.height
            grid = self.grid.slice(topX, topY, self.width, self.height, rot_k=0)
            vis_mask = np.ones((self.width, self.height), dtype=bool)
        else:
            topX, topY, botX, botY = agent.get_view_exts()
            grid = self.grid.slice(
                topX, topY, agent.view_size, agent.view_size, rot_k=agent.dir + 1
            )
            # Process occluders and visibility
            # Note that this incurs some slight performance cost
            vis_mask = agent.process_vis(grid.opacity)

        # Warning about the rest of the function:
        #  Allows masking away objects that the agent isn't supposed to see.
        #  But breaks consistency between the states of the grid objects in the parial views
        #   and the grid objects overall.
        if len(getattr(agent, "hide_item_types", [])) > 0:
            for i in range(grid.width):
                for j in range(grid.height):
                    item = grid.get(i, j)
                    if (
                        (item is not None)
                        and (item is not agent)
                        and (item.type in agent.hide_item_types)
                    ):
                        if len(item.agents) > 0:
                            grid.set(i, j, item.agents[0])
                        else:
                            grid.set(i, j, None)

        return grid, vis_mask

    def render(
        self,
        mode="human",
        close=False,
        highlight=True,
        tile_size=TILE_PIXELS,
        show_agent_views=True,
        max_agents_per_col=3,
        agent_col_width_frac=0.3,
        agent_col_padding_px=2,
        pad_grey=100,
    ):
        """Render the whole-grid human view"""

        if close:
            if self.window:
                self.window.close()
            return

        if mode == "human" and not self.window:
            self.window = SimpleImageViewer(caption="Marlgrid")

        # Compute which cells are visible to the agent
        highlight_mask = np.full((self.width, self.height), False, dtype=np.bool)
        for agent in self.agents:
            if agent.active:
                if self._full_obs:
                    xlow, ylow, xhigh, yhigh = 0, 0, self.width, self.height
                else:
                    xlow, ylow, xhigh, yhigh = agent.get_view_exts()

                dxlow, dylow = max(0, 0 - xlow), max(0, 0 - ylow)
                dxhigh, dyhigh = max(0, xhigh - self.grid.width), max(
                    0, yhigh - self.grid.height
                )
                if agent.see_through_walls:
                    highlight_mask[
                        xlow + dxlow : xhigh - dxhigh, ylow + dylow : yhigh - dyhigh
                    ] = True
                else:
                    a, b = self.gen_obs_grid(agent)
                    highlight_mask[
                        xlow + dxlow : xhigh - dxhigh, ylow + dylow : yhigh - dyhigh
                    ] |= rotate_grid(b, a.orientation)[
                        dxlow : (xhigh - xlow) - dxhigh, dylow : (yhigh - ylow) - dyhigh
                    ]

        # Render the whole grid
        img = self.grid.render(
            tile_size, highlight_mask=highlight_mask if highlight else None
        )
        rescale = lambda X, rescale_factor=2: np.kron(
            X, np.ones((int(rescale_factor), int(rescale_factor), 1))
        )

        if show_agent_views:

            target_partial_width = int(
                img.shape[0] * agent_col_width_frac - 2 * agent_col_padding_px
            )
            target_partial_height = (
                img.shape[1] - 2 * agent_col_padding_px
            ) // max_agents_per_col

            agent_views = [self.gen_agent_obs(agent) for agent in self.agents]
            agent_views = [
                view["pov"] if isinstance(view, dict) else view for view in agent_views
            ]
            agent_views = [
                rescale(
                    view,
                    min(
                        target_partial_width / view.shape[0],
                        target_partial_height / view.shape[1],
                    ),
                )
                for view in agent_views
            ]
            agent_views = [
                agent_views[pos : pos + max_agents_per_col]
                for pos in range(0, len(agent_views), max_agents_per_col)
            ]

            f_offset = (
                lambda view: np.array(
                    [
                        target_partial_height - view.shape[1],
                        target_partial_width - view.shape[0],
                    ]
                )
                // 2
            )

            cols = []
            for col_views in agent_views:
                col = np.full(
                    (img.shape[0], target_partial_width + 2 * agent_col_padding_px, 3),
                    pad_grey,
                    dtype=np.uint8,
                )
                for k, view in enumerate(col_views):
                    offset = f_offset(view) + agent_col_padding_px
                    offset[0] += k * target_partial_height
                    col[
                        offset[0] : offset[0] + view.shape[0],
                        offset[1] : offset[1] + view.shape[1],
                        :,
                    ] = view
                cols.append(col)

            img = np.concatenate((img, *cols), axis=1)

        if mode == "human":
            if not self.window.isopen:
                self.window.imshow(img)
                self.window.window.set_caption("Marlgrid")
            else:
                self.window.imshow(img)

        return img
