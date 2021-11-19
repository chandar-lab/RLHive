import random
from collections import deque

import numpy as np
import torch
import yaml


def load_config(args):
    """Used to load config for experiments. Agents, environment, and loggers components
    in main config file can be overrided based on other log files.
    """
    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.agent_config is not None:
        with open(args.agent_config) as f:
            config["agents"] = yaml.safe_load(f)
    if args.env_config is not None:
        with open(args.env_config) as f:
            config["environment"] = yaml.safe_load(f)
    if args.logger_config is not None:
        with open(args.logger_config) as f:
            config["loggers"] = yaml.safe_load(f)
    return config


def set_seed(seed):
    """This reduces some sources of randomness in experiments. To get reproducible
    results, you must run on the same machine and set the environment variable
    CUBLAS_WORKSPACE_CONFIG to ":4096:8" or ":16:8" before starting the experiment.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


class Metrics:
    """Class used to keep track of separate metrics for each agent as well general
    episode metrics.
    """

    def __init__(self, agents, agent_metrics, episode_metrics):
        """Initialise Metrics object.

        Args:
            agents: List of agents for which object will track metrics.
            agent_metrics: List of metrics to track for each agent. Should be a list of
                tuples (metric_name, metric_init) where metric_init is either the
                initial value of the metric or a callable with no arguments that
                creates the initial metric.
            episode_metrics: List of non agent specific metrics to keep track of.
                Should be a list of tuples (metric_name, metric_init) where metric_init
                is either the initial value of the metric or a callable with no
                arguments that creates the initial metric.
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

    def __init__(self, agents, stack_size):
        """Constructor for TransitionInfo object.

        Args:
            agents: list of agents that will be kept track of.
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
        """Check if agent has started its episode."""
        return self._started[agent.id]

    def start_agent(self, agent):
        """Set the agent's start flag to true."""
        self._started[agent.id] = True

    def record_info(self, agent, info):
        """Update some information for the agent."""
        self._transitions[agent.id].update(info)
        if "observation" in info:
            self._previous_observations[agent.id].append(info["observation"])

    def update_reward(self, agent, reward):
        """Add a reward to the agent."""
        self._transitions[agent.id]["reward"] += reward

    def update_all_rewards(self, rewards):
        """Update the rewards for all agents. If rewards is list, it updates the rewards
        according to the order of agents provided in the initializer. If rewards is a
        dict, the keys should be the agent ids for the agents and the values should be
        the rewards for those agents. If rewards is a float or int, every agent is
        updated with that reward.
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
        """
        info = self._transitions[agent.id]
        info["done"] = done
        self._transitions[agent.id] = {"reward": 0.0}
        return info

    def get_stacked_state(self, agent, observation):
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
    if len(xs) == 0:
        return np.array([])

    if isinstance(xs[0], dict):
        return {k: np.concatenate([x[k] for x in xs], axis=0) for k in xs[0]}
    else:
        return np.concatenate(xs, axis=0)
