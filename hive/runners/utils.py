import yaml
import numpy as np


def load_config(args):
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


class Metrics:
    def __init__(self, agents, agent_metrics, episode_metrics):
        self._metrics = {}
        self._agent_metrics = agent_metrics
        self._episode_metrics = episode_metrics
        self._agent_ids = [agent.id for agent in agents]
        self.reset_metrics()

    def reset_metrics(self):
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


class TransitionInfo:
    def __init__(self, agents):
        self._agent_ids = [agent.id for agent in agents]
        self._num_agents = len(agents)
        self.reset()

    def reset(self):
        self._transitions = {agent_id: {"reward": 0.0} for agent_id in self._agent_ids}
        self._started = {agent_id: False for agent_id in self._agent_ids}

    def is_started(self, agent):
        return self._started[agent.id]

    def start_agent(self, agent):
        self._started[agent.id] = True

    def record_info(self, agent, info):
        self._transitions[agent.id].update(info)

    def update_reward(self, agent, value):
        self._transitions[agent.id]["reward"] += value

    def update_all_rewards(self, values):
        if isinstance(values, list) or isinstance(values, np.ndarray):
            for idx, agent_id in enumerate(self._agent_ids):
                self._transitions[agent_id]["reward"] += values[idx]
        elif isinstance(values, int) or isinstance(values, float):
            for agent_id in self._agent_ids:
                self._transitions[agent_id]["reward"] += values
        else:
            for agent_id in values:
                self._transitions[agent_id]["reward"] += values[agent_id]

    def get_info(self, agent, done=False):
        info = self._transitions[agent.id]
        info["done"] = done
        self._transitions[agent.id] = {"reward": 0.0}
        return info
