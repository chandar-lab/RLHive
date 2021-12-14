import argparse
import copy

from hive import agents as agent_lib
from hive import envs
from hive.runners.base import Runner
from hive.runners.utils import TransitionInfo, load_config
from hive.utils import experiment, loggers, schedule, utils
from hive.utils.registry import get_parsed_args


class MultiAgentRunner(Runner):
    """Runner class used to implement a multiagent training loop."""

    def __init__(
        self,
        environment,
        agents,
        logger,
        experiment_manager,
        train_steps,
        test_frequency,
        test_episodes,
        stack_size,
        self_play,
        max_steps_per_episode=27000,
    ):
        """Initializes the Runner object.

        Args:
            environment (BaseEnv): Environment used in the training loop.
            agents (list[Agent]): List of agents that interact with the environment
            logger (ScheduledLogger): Logger object used to log metrics.
            experiment_manager (Experiment): Experiment object that saves the state of
                the training.
            train_steps (int): How many steps to train for. If this is -1, there is no
                limit for the number of training steps.
            test_frequency (int): After how many training steps to run testing
                episodes. If this is -1, testing is not run.
            test_episodes (int): How many episodes to run testing for.
            stack_size (int): The number of frames in an observation sent to an agent.
            max_steps_per_episode (int): The maximum number of steps to run an episode
                for.
        """
        super().__init__(
            environment,
            agents,
            logger,
            experiment_manager,
            train_steps,
            test_frequency,
            test_episodes,
            max_steps_per_episode,
        )
        self._transition_info = TransitionInfo(self._agents, stack_size)
        self._self_play = self_play

    def run_one_step(self, observation, turn, episode_metrics):
        """Run one step of the training loop.

        If it is the agent's first turn during the episode, do not run an update step.
        Otherwise, run an update step based on the previous action and accumulated
        reward since then.

        Args:
            observation: Current observation that the agent should create an action
                for.
            turn (int): Agent whose turn it is.
            episode_metrics (Metrics): Keeps track of metrics for current episode.
        """
        super().run_one_step(observation, turn, episode_metrics)
        agent = self._agents[turn]
        if self._transition_info.is_started(agent):
            info = self._transition_info.get_info(agent)
            if self._training:
                agent.update(copy.deepcopy(info))

            episode_metrics[agent.id]["reward"] += info["reward"]
            episode_metrics[agent.id]["episode_length"] += 1
            episode_metrics["full_episode_length"] += 1
        else:
            self._transition_info.start_agent(agent)

        stacked_observation = self._transition_info.get_stacked_state(
            agent, observation
        )
        action = agent.act(stacked_observation)
        next_observation, reward, done, turn, other_info = self._environment.step(
            action
        )
        self._transition_info.record_info(
            agent,
            {
                "observation": observation,
                "action": action,
                "info": other_info,
            },
        )
        if self._self_play:
            self._transition_info.record_info(
                agent,
                {
                    "agent_id": agent.id,
                },
            )
        self._transition_info.update_all_rewards(reward)
        return done, next_observation, turn

    def run_end_step(self, episode_metrics, done=True):
        """Run the final step of an episode.

        After an episode ends, iterate through agents and update then with the final
        step in the episode.

        Args:
            episode_metrics (Metrics): Keeps track of metrics for current episode.
            done (bool): Whether this step was terminal.

        """
        for agent in self._agents:
            if self._transition_info.is_started(agent):
                info = self._transition_info.get_info(agent, done=done)

                if self._training:
                    agent.update(info)
                episode_metrics[agent.id]["episode_length"] += 1
                episode_metrics["full_episode_length"] += 1
            episode_metrics[agent.id]["reward"] += info["reward"]

    def run_episode(self):
        """Run a single episode of the environment."""
        episode_metrics = self.create_episode_metrics()
        done = False
        observation, turn = self._environment.reset()
        self._transition_info.reset()
        steps = 0
        # Run the loop until the episode ends or times out
        while not done and steps < self._max_steps_per_episode:
            done, observation, turn = self.run_one_step(
                observation, turn, episode_metrics
            )
            steps += 1

        # Run the final update.
        self.run_end_step(episode_metrics, done)
        return episode_metrics


def set_up_experiment(config):
    """Returns a :py:class:`MultiAgentRunner` object based on the config and any
    command line arguments.

    Args:
        config: Configuration for experiment.
    """

    # Parses arguments from the command line.
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
            "self_play": bool,
            "num_agents": int,
        }
    )
    config.update(args)
    full_config = utils.Chomp(copy.deepcopy(config))

    if "seed" in config:
        utils.seeder.set_global_seed(config["seed"])

    # Set up environment
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

    # Set up agents
    agents = []
    full_config["agents"] = []
    num_agents = config["num_agents"] if config["self_play"] else len(config["agents"])
    for idx in range(num_agents):

        if not config["self_play"] or idx == 0:
            agent_config = config["agents"][idx]
            if config.get("stack_size", 1) > 1:
                agent_config["kwargs"]["obs_dim"] = (
                    config["stack_size"] * env_spec.obs_dim[idx][0],
                    *env_spec.obs_dim[idx][1:],
                )
            else:
                agent_config["kwargs"]["obs_dim"] = env_spec.obs_dim[idx]
            agent_config["kwargs"]["act_dim"] = env_spec.act_dim[idx]
            agent_config["kwargs"]["logger"] = logger

            if "replay_buffer" in agent_config["kwargs"]:
                replay_args = agent_config["kwargs"]["replay_buffer"]["kwargs"]
                replay_args["observation_shape"] = env_spec.obs_dim[idx]
            agent, full_agent_config = agent_lib.get_agent(
                agent_config, f"agents.{idx}"
            )
            agents.append(agent)
            full_config["agents"].append(full_agent_config)
        else:
            agents.append(copy.copy(agents[0]))
            agents[-1]._id = idx

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
        agents=agents,
    )

    # Set up runner
    runner = MultiAgentRunner(
        environment,
        agents,
        logger,
        experiment_manager,
        config.get("train_steps", -1),
        config.get("test_frequency", -1),
        config.get("test_episodes", 1),
        config.get("stack_size", 1),
        config.get("self_play", False),
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
