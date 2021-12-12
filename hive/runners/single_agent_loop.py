import argparse
import copy

from hive import agents as agent_lib
from hive import envs
from hive.runners.base import Runner
from hive.runners.utils import TransitionInfo, load_config
from hive.utils import experiment, loggers, schedule, utils
from hive.utils.registry import get_parsed_args


class SingleAgentRunner(Runner):
    """Runner class used to implement a sinle-agent training loop."""

    def __init__(
        self,
        environment,
        agent,
        logger,
        experiment_manager,
        train_steps,
        test_frequency,
        test_episodes,
        stack_size,
        max_steps_per_episode=27000,
    ):
        """Initializes the Runner object.

        Args:
            environment (BaseEnv): Environment used in the training loop.
            agent (Agent): Agent that will interact with the environment
            logger (ScheduledLogger): Logger object used to log metrics.
            experiment_manager (Experiment): Experiment object that saves the state of
                the training.
            train_steps (int): How many steps to train for. If this is -1, there is no
                limit for the number of training steps.
            test_frequency (int): After how many training steps to run testing
                episodes. If this is -1, testing is not run.
            test_episodes (int): How many episodes to run testing for duing each test
                phase.
            stack_size (int): The number of frames in an observation sent to an agent.
            max_steps_per_episode (int): The maximum number of steps to run an episode
                for.
        """
        super().__init__(
            environment,
            agent,
            logger,
            experiment_manager,
            train_steps,
            test_frequency,
            test_episodes,
            max_steps_per_episode,
        )
        self._transition_info = TransitionInfo(self._agents, stack_size)

    def run_one_step(self, observation, episode_metrics):
        """Run one step of the training loop.

        Args:
            observation: Current observation that the agent should create an action
                for.
            episode_metrics (Metrics): Keeps track of metrics for current episode.
        """
        super().run_one_step(observation, 0, episode_metrics)
        agent = self._agents[0]
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
        if self._training:
            agent.update(copy.deepcopy(info))

        self._transition_info.record_info(agent, info)
        episode_metrics[agent.id]["reward"] += info["reward"]
        episode_metrics[agent.id]["episode_length"] += 1
        episode_metrics["full_episode_length"] += 1

        return done, next_observation

    def run_episode(self):
        """Run a single episode of the environment."""
        episode_metrics = self.create_episode_metrics()
        done = False
        observation, _ = self._environment.reset()
        self._transition_info.reset()
        self._transition_info.start_agent(self._agents[0])
        steps = 0
        # Run the loop until the episode ends or times out
        while not done and steps < self._max_steps_per_episode:
            done, observation = self.run_one_step(observation, episode_metrics)
            steps += 1

        return episode_metrics


def set_up_experiment(config):
    """Returns a :py:class:`SingleAgentRunner` object based on the config and any
    command line arguments.

    Args:
        config: Configuration for experiment.
    """

    args = get_parsed_args(
        {
            "seed": int,
            "train_steps": int,
            "test_frequency": int,
            "test_episodes": int,
            "max_steps_per_episode": int,
            "stack_size": int,
            "resume": bool,
            "run_name": str,
            "save_dir": str,
        }
    )
    config.update(args)
    full_config = utils.Chomp(copy.deepcopy(config))
    if "seed" in config:
        utils.seeder.set_global_seed(config["seed"])

    environment, full_config["environment"] = envs.get_env(
        config["environment"], "environment"
    )
    env_spec = environment.env_spec

    # Set up loggers
    logger_config = config.get("loggers", {"name": "NullLogger"})
    if logger_config is None or len(logger_config) == 0:
        logger_config = {"name": "NullLogger"}
    if isinstance(logger_config, list):
        logger_config = {
            "name": "CompositeLogger",
            "kwargs": {"logger_list": logger_config},
        }

    logger, full_config["loggers"] = loggers.get_logger(logger_config, "loggers")

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
        config.get("test_frequency", -1),
        config.get("test_episodes", 1),
        config.get("stack_size", 1),
        config.get("max_steps_per_episode", 1e9),
    )
    if config.get("resume", False):
        runner.resume()

    return runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    parser.add_argument("-p", "--preset-config")
    parser.add_argument("-a", "--agent-config")
    parser.add_argument("-e", "--env-config")
    parser.add_argument("-l", "--logger-config")
    args, _ = parser.parse_known_args()
    if args.config is None and args.preset_config is None:
        raise ValueError("Config needs to be provided")
    config = load_config(
        args.config,
        args.preset_config,
        args.agent_config,
        args.env_config,
        args.logger_config,
    )
    runner = set_up_experiment(config)
    runner.run_training()


if __name__ == "__main__":
    main()
