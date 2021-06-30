import argparse
import copy
from hive.utils.registry import get_parsed_args

from hive import agents as agent_lib
from hive import envs
from hive.runners.base import Runner
from hive.runners.utils import TransitionInfo, load_config, set_seed
from hive.utils import experiment, logging, schedule, utils


class SingleAgentRunner(Runner):
    """Runner class used to implement a sinle-agent training loop."""

    def __init__(
        self,
        environment,
        agents,
        logger,
        experiment_manager,
        train_steps,
        train_episodes,
        test_frequency,
        test_num_episodes,
        stack_size,
    ):
        super().__init__(
            environment,
            agents,
            logger,
            experiment_manager,
            train_steps,
            train_episodes,
            test_frequency,
            test_num_episodes,
        )
        self._transition_info = TransitionInfo(self._agents, stack_size)

    def run_one_step(self, observation, turn, episode_metrics):
        """Run one step of the training loop.

        If it is the agent's first turn during the episode, do not run an update step.
        Otherwise, run an update step based on the previous action and accumulated
        reward since then.

        Args:
            observation: Current observation that the agent should create an action for.
            turn: Agent whose turn it is.
            episode_metrics: Metrics object keeping track of metrics for current episode.
        """
        agent = self._agents[turn]
        stacked_observation = self._transition_info.get_stacked_state(
            agent, observation
        )
        action = agent.act(stacked_observation)
        next_observation, reward, done, _, other_info = self._environment.step(action)

        info = {
            "observation": observation,
            "reward": reward,
            "action": action,
            "done": done,
            "info": other_info,
        }
        agent.update(info)
        self._transition_info.record_info(agent, info)
        episode_metrics[agent.id]["reward"] += info["reward"]
        episode_metrics[agent.id]["episode_length"] += 1
        episode_metrics["full_episode_length"] += 1

        return done, next_observation, _

    def run_episode(self):
        """Run a single episode of the environment."""
        episode_metrics = self.create_episode_metrics()
        done = False
        observation, _ = self._environment.reset()
        self._transition_info.reset()
        self._transition_info.start_agent(self._agents[0])

        # Run the loop until either training ends or the episode ends
        while self._train_step_schedule.get_value() and not done:
            done, observation, _ = self.run_one_step(observation, 0, episode_metrics)
            self._train_step_schedule.update()

        return episode_metrics


def set_up_experiment(config):
    """Returns a runner object based on the config."""

    args = get_parsed_args(
        {
            "seed": int,
            "train_steps": int,
            "train_episodes": int,
            "test_frequency": int,
            "test_num_episodes": int,
            "stack_size": int,
            "resume": bool,
            "run_name": str,
            "save_dir": str,
        }
    )
    config.update(args)
    full_config = utils.Chomp(copy.deepcopy(config))
    if "seed" in config:
        set_seed(config["seed"])

    environment, full_config["environment"] = envs.get_env(config["environment"], "env")
    env_spec = environment.env_spec

    # Set up loggers
    logger_config = config.get("loggers", {"name": "NullLogger"})
    if logger_config is None or len(logger_config) == 0:
        logger_config = {"name": "NullLogger"}
    if isinstance(logger_config, list):
        for logger in logger_config:
            logger["kwargs"] = logger.get("kwargs", {})
            logger["kwargs"]["timescales"] = ["train_episodes", "test_episodes"]
        logger_config = {
            "name": "CompositeLogger",
            "kwargs": {"logger_list": logger_config},
        }
    else:
        logger_config["kwargs"] = logger_config.get("kwargs", {})
        logger_config["kwargs"]["timescales"] = ["train_episodes", "test_episodes"]

    logger, full_config["loggers"] = logging.get_logger(logger_config, "loggers")

    # Set up agent
    if config.get("stack_size", 1) > 1:
        config["agent"]["kwargs"]["obs_dim"] = (
            config["stack_size"] * env_spec.obs_dim[0][0],
            *env_spec.obs_dim[0][1:],
        )
    else:
        config["agent"]["kwargs"]["obs_dim"] = env_spec.obs_dim[0]
    config["agent"]["kwargs"]["act_dim"] = env_spec.act_dim[0]
    config["agent"]["kwargs"]["logger"] = logger
    if "replay_buffer" in config["agent"]["kwargs"]:
        replay_args = config["agent"]["kwargs"]["replay_buffer"]["kwargs"]
        replay_args["observation_shape"] = env_spec.obs_dim[0]
    agent, full_config["agent"] = agent_lib.get_agent(config["agent"], "agent")

    # Set up experiment manager
    saving_schedule, full_config["saving_schedule"] = schedule.get_schedule(
        config["saving_schedule"], "saving_schedule"
    )
    experiment_manager = experiment.Experiment(
        config["run_name"], config["save_dir"], saving_schedule
    )
    experiment_manager.register_experiment(
        config=full_config,
        logger=logger,
        agents=agent,
    )
    # Set up runner
    runner = SingleAgentRunner(
        environment,
        agent,
        logger,
        experiment_manager,
        config.get("train_steps", -1),
        config.get("train_episodes", -1),
        config.get("test_frequency", -1),
        config.get("test_num_episodes", 1),
        config.get("stack_size", 1),
    )
    if config.get("resume", False):
        runner.resume()

    return runner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="./config.yml")
    parser.add_argument("-a", "--agent-config")
    parser.add_argument("-e", "--env-config")
    parser.add_argument("-l", "--logger-config")
    args, _ = parser.parse_known_args()
    config = load_config(args)
    runner = set_up_experiment(config)
    runner.run_training()

if __name__ == "__main__":
    main()
