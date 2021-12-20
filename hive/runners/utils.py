import os
from collections import deque

import numpy as np
import torch
import yaml

from hive.utils.utils import PACKAGE_ROOT


def load_config(
    config=None,
    preset_config=None,
    agent_config=None,
    env_config=None,
    logger_config=None,
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
    else:
        with open(os.path.join(PACKAGE_ROOT, "configs", preset_config)) as f:
            yaml_config = yaml.safe_load(f)
    if agent_config is not None:
        with open(agent_config) as f:
            if "agents" in yaml_config:
                yaml_config["agents"] = yaml.safe_load(f)
            else:
                yaml_config["agent"] = yaml.safe_load(f)
    if env_config is not None:
        with open(env_config) as f:
            yaml_config["environment"] = yaml.safe_load(f)
    if logger_config is not None:
        with open(logger_config) as f:
            yaml_config["loggers"] = yaml.safe_load(f)
    return yaml_config


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

    This object also handles padding and stacking observations for agents.
    """

    def __init__(self, agents, stack_size):
        """
        Args:
            agents (list[Agent]): list of agents that will be kept track of.
            stack_size (int): How many observations will be stacked.
        """
        self._agent_ids = [agent.id for agent in agents]
        self._num_agents = len(agents)
        self._stack_size = stack_size
        self.reset()

    def reset(self):
        """Reset the object by clearing all info."""
        self._transitions = {agent_id: {"reward": 0.0} for agent_id in self._agent_ids}
        self._started = {agent_id: False for agent_id in self._agent_ids}
        self._previous_observations = {
            agent_id: deque(maxlen=self._stack_size - 1) for agent_id in self._agent_ids
        }

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
        if "observation" in info:
            self._previous_observations[agent.id].append(info["observation"])

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

    def get_info(self, agent, done=False):
        """Get all the info for the agent, and reset the info for that agent. Also adds
        a done value to the info dictionary that is based on the done parameter to the
        function.

        Args:
            agent (Agent): Agent to get transition update info for.
            done (bool): Whether this transition is terminal.
        """
        info = self._transitions[agent.id]
        info["done"] = done
        self._transitions[agent.id] = {"reward": 0.0}
        return info

    def get_stacked_state(self, agent, observation):
        """Create a stacked state for the agent. The previous observations recorded
        by this agent are stacked with the current observation. If not enough
        observations have been recorded, zero arrays are appended.

        Args:
            agent (Agent): Agent to get stacked state for.
            observation: Current observation.
        """

        if self._stack_size == 1:
            return observation
        while len(self._previous_observations[agent.id]) < self._stack_size - 1:
            self._previous_observations[agent.id].append(zeros_like(observation))

        stacked_observation = concatenate(
            list(self._previous_observations[agent.id]) + [observation]
        )
        return stacked_observation

    def __repr__(self):
        return str(
            {
                "transtions": self._transitions,
                "started": self._started,
                "previous_observations": self._previous_observations,
            }
        )


def zeros_like(x):
    """Create a zero state like some state. This handles slightly more complex
    objects such as lists and dictionaries of numpy arrays and torch Tensors.

    Args:
        x (np.ndarray | torch.Tensor | dict | list): State used to define
            structure/state of zero state.
    """
    if isinstance(x, np.ndarray):
        return np.zeros_like(x)
    elif isinstance(x, torch.Tensor):
        return torch.zeros_like(x)
    elif isinstance(x, dict):
        return {k: zeros_like(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [zeros_like(item) for item in x]
    else:
        return 0


def concatenate(xs):
    """Concatenates numpy arrays or dictionaries of numpy arrays.

    Args:
        xs (list): List of objects to concatenate.
    """

    if len(xs) == 0:
        return np.array([])

    if isinstance(xs[0], dict):
        return {k: np.concatenate([x[k] for x in xs], axis=0) for k in xs[0]}
    else:
        return np.concatenate(xs, axis=0)
