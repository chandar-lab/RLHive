import os
import sys
from collections import deque
from typing import Optional

import numpy as np
import yaml

from hive.utils.config import dict_to_config
from hive.utils.utils import PACKAGE_ROOT


def load_config(
    config: Optional[str] = None,
    preset_config: Optional[str] = None,
    agent_config: Optional[str] = None,
    env_config: Optional[str] = None,
    logger_config: Optional[str] = None,
):
    """Used to load config for experiments. Agents, environment, and loggers components
    in main config file can be overrided based on other log files.

    Args:
        config (str): Path to configuration file. Either this or :obj:`preset_config`
            must be passed.
        preset_config (str): Path to a preset hive config. This path should be relative
            to :obj:`hive/configs`. For example, the Atari DQN config would be
            :obj:`atari/dqn.yml`.
        agent_config (str): Path to agent configuration file. Overrides settings in
            base config.
        env_config (str): Path to environment configuration file. Overrides settings in
            base config.
        logger_config (str): Path to logger configuration file. Overrides settings in
            base config.
    """
    if config is not None:
        with open(config) as f:
            yaml_config = yaml.safe_load(f)
    elif preset_config is not None:
        with (PACKAGE_ROOT / "configs" / preset_config).open() as f:
            yaml_config = yaml.safe_load(f)
    else:
        raise ValueError("Either config or preset_config must be passed.")
    if agent_config is not None:
        with open(agent_config) as f:
            if "agents" in yaml_config["kwargs"]:
                yaml_config["kwargs"]["agents"] = yaml.safe_load(f)
            else:
                yaml_config["kwargs"]["agent"] = yaml.safe_load(f)
    if env_config is not None:
        with open(env_config) as f:
            yaml_config["kwargs"]["environment"] = yaml.safe_load(f)
    if logger_config is not None:
        with open(logger_config) as f:
            yaml_config["kwargs"]["loggers"] = yaml.safe_load(f)

    return dict_to_config(yaml_config)


class Metrics:
    """Class used to keep track of separate metrics for each agent as well general
    episode metrics.
    """

    def __init__(self, agents, agent_metrics, episode_metrics):
        """
        Args:
            agents (list[Agent]): List of agents for which object will track metrics.
            agent_metrics (list[(str, (callable | obj))]): List of metrics to track
                for each agent. Should be a list of tuples (metric_name, metric_init)
                where metric_init is either the initial value of the metric or a
                callable that takes no arguments and creates the initial metric.
            episode_metrics (list[(str, (callable | obj))]): List of non agent specific
                metrics to keep track of. Should be a list of tuples
                (metric_name, metric_init) where metric_init is either the initial
                value of the metric or a callable with no arguments that creates the
                initial metric.
        """
        self._metrics = {}
        self._agent_metrics = agent_metrics
        self._episode_metrics = episode_metrics
        self._agent_ids = [agent.id for agent in agents]
        self.reset_metrics()

    def reset_metrics(self):
        """Resets all metrics to their initial values."""
        for agent_id in self._agent_ids:
            self._metrics[agent_id] = {}
            for metric_name, metric_value in self._agent_metrics:
                self._metrics[agent_id][metric_name] = (
                    metric_value() if callable(metric_value) else metric_value
                )

        for metric_name, metric_value in self._episode_metrics:
            self._metrics[metric_name] = (
                metric_value() if callable(metric_value) else metric_value
            )

    def get_flat_dict(self):
        """Get a flat dictionary version of the metrics. Each agent metric will be
        prefixed by the agent id.
        """
        metrics = {}
        for metric, _ in self._episode_metrics:
            metrics[metric] = self._metrics[metric]
        for agent_id in self._agent_ids:
            for metric, _ in self._agent_metrics:
                metrics[f"{agent_id}_{metric}"] = self._metrics[agent_id][metric]
        return metrics

    def __getitem__(self, key):
        return self._metrics[key]

    def __setitem__(self, key, value):
        self._metrics[key] = value

    def __repr__(self) -> str:
        return str(self._metrics)


class TransitionInfo:
    """Used to keep track of the most recent transition for each agent.

    Any info that the agent needs to remember for updating can be stored here. Should
    be completely reset between episodes. After any info is extracted, it is
    automatically removed from the object. Also keeps track of which agents have
    started their episodes.
    """

    def __init__(self, agents):
        """
        Args:
            agents (list[Agent]): list of agents that will be kept track of.
        """
        self._agent_ids = [agent.id for agent in agents]
        self._num_agents = len(agents)
        self.reset()

    def reset(self):
        """Reset the object by clearing all info."""
        self._transitions = {agent_id: {"reward": 0.0} for agent_id in self._agent_ids}
        self._started = {agent_id: False for agent_id in self._agent_ids}

    def is_started(self, agent):
        """Check if agent has started its episode.

        Args:
            agent (Agent): Agent to check.
        """
        return self._started[agent.id]

    def start_agent(self, agent):
        """Set the agent's start flag to true.

        Args:
            agent (Agent): Agent to start.
        """
        self._started[agent.id] = True

    def record_info(self, agent, info):
        """Update some information for the agent.

        Args:
            agent (Agent): Agent to update.
            info (dict): Info to add to the agent's state.
        """
        self._transitions[agent.id].update(info)

    def update_reward(self, agent, reward):
        """Add a reward to the agent.

        Args:
            agent (Agent): Agent to update.
            reward (float): Reward to add to agent.
        """
        self._transitions[agent.id]["reward"] += reward

    def update_all_rewards(self, rewards):
        """Update the rewards for all agents. If rewards is list, it updates the rewards
        according to the order of agents provided in the initializer. If rewards is a
        dict, the keys should be the agent ids for the agents and the values should be
        the rewards for those agents. If rewards is a float or int, every agent is
        updated with that reward.

        Args:
            rewards (float | list | np.ndarray | dict): Rewards to update agents with.
        """
        if isinstance(rewards, list) or isinstance(rewards, np.ndarray):
            for idx, agent_id in enumerate(self._agent_ids):
                self._transitions[agent_id]["reward"] += rewards[idx]
        elif isinstance(rewards, int) or isinstance(rewards, float):
            for agent_id in self._agent_ids:
                self._transitions[agent_id]["reward"] += rewards
        else:
            for agent_id in rewards:
                self._transitions[agent_id]["reward"] += rewards[agent_id]

    def get_info(self, agent, terminated=False, truncated=False):
        """Get all the info for the agent, and reset the info for that agent. Also adds
        a done value to the info dictionary that is based on the done parameter to the
        function.

        Args:
            agent (Agent): Agent to get transition update info for.
            done (bool): Whether this transition is terminal.
        """
        info = self._transitions[agent.id]
        info["terminated"] = terminated
        info["truncated"] = truncated
        self._transitions[agent.id] = {"reward": 0.0}
        return info

    def __repr__(self):
        return str(
            {
                "transitions": self._transitions,
                "started": self._started,
            }
        )
