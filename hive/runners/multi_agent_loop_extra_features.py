import argparse
import copy

from hive import agents as agent_lib
from hive import envs
from hive.utils import experiment, logging, schedule, utils
from hive.runners.utils import load_config, TransitionInfo, set_seed
from hive.runners.base import Runner
from hive.utils.registry import get_parsed_args
from hive.runners.utils import Metrics

from statistics import mean

class MultiAgentRunner(Runner):
    """Runner class used to implement a multiagent training loop."""

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
        """Initializes the Runner object.
        Args:
            environment: Environment used in the training loop.
            agents: List of agents that interact with the environment
            logger: Logger object used to log metrics.
            experiment_manager: ExperimentManager object that saves the state of the
                training.
            train_steps: How many steps to train for. If this is -1, there is no limit
                for the number of training steps. If both this and train_episodes are
                -1, training loop will not terminate.
            train_episodes: How many episodes to train for. If this is -1, there is no
                limit for the number of training episodes. If both this and train_steps
                are -1, training loop will not terminate.
            test_frequency: After how many training episodes to run testing episodes.
                If this is -1, testing is not run.
            test_num_episodes: How many testing episodes to run during each testing
                period.
            stack_size: The number of frames in an observation sent to an agent.
        """
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
        self._environment = environment
        self._agents = agents
        self._logger = logger
        self._experiment_manager = experiment_manager
        if train_steps == -1:
            self._train_step_schedule = schedule.ConstantSchedule(True)
        else:
            self._train_step_schedule = schedule.SwitchSchedule(
                True, False, train_steps
            )
        if train_episodes == -1:
            self._train_episode_schedule = schedule.ConstantSchedule(True)
        else:
            self._train_episode_schedule = schedule.SwitchSchedule(
                True, False, train_episodes
            )
        if test_frequency == -1:
            self._test_schedule = schedule.ConstantSchedule(False)
        else:
            self._test_schedule = schedule.DoublePeriodicSchedule(
                False, True, test_frequency, test_num_episodes
            )
        self._train_step_schedule.update()
        self._test_schedule.update()
        self._experiment_manager.experiment_state.add_from_dict(
            {
                "train_step_schedule": self._train_step_schedule,
                "train_episode_schedule": self._train_episode_schedule,
                "test_schedule": self._test_schedule,
            }
        )

        self._transition_info = TransitionInfo(self._agents, stack_size)
        self._training = True

    def create_episode_metrics(self):
        """Create the metrics used during the loop."""
        return Metrics(
            self._agents,
            [("reward", 0), ("disc_reward", 0), ("episode_length", 0)],
            [("full_episode_length", 0), ("env_steps", -1)],
        )

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
        if self._transition_info.is_started(agent):
            info = self._transition_info.get_info(agent)
            agent.update(info)
            episode_metrics[agent.id]["reward"] += info["reward"]
            episode_metrics[agent.id]["disc_reward"] += agent._discount_rate ** (episode_metrics[agent.id]["episode_length"]) * info["reward"]
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
        self._transition_info.update_all_rewards(reward)
        return done, next_observation, turn

    def run_end_step(self, episode_metrics):
        """Run the final step of an episode.

        After an episode ends, iterate through agents and update then with the final
        step in the episode.

        Args:
            episode_metrics: Metrics object keeping track of metrics for current episode.

        """
        for agent in self._agents:
            info = self._transition_info.get_info(agent, done=True)
            agent.update(info)
            episode_metrics[agent.id]["reward"] += info["reward"]
            episode_metrics[agent.id]["disc_reward"] += agent._discount_rate ** (episode_metrics[agent.id]["episode_length"]) * info["reward"]
            episode_metrics[agent.id]["episode_length"] += 1
            episode_metrics["full_episode_length"] += 1

    def run_episode(self):
        """Run a single episode of the environment."""
        self._transition_info.reset()
        episode_metrics = self.create_episode_metrics()
        done = False
        observation, turn = self._environment.reset()

        # Run the loop until either training ends or the episode ends
        while (
            not self._training or self._train_step_schedule.get_value()
        ) and not done:
            done, observation, turn = self.run_one_step(
                observation, turn, episode_metrics
            )
            if self._training:
                self._train_step_schedule.update()

        # If the episode ended, run the final update.
        if done:
            self.run_end_step(episode_metrics)
        return episode_metrics

    def run_training(self):
        """Run the training loop."""

        while (
            self._train_episode_schedule.update()
            and self._train_step_schedule.get_value()
        ):
            # Run training episode
            self.train_mode(True)
            episode_metrics = self.run_episode()

            # Run test episodes
            episode_metrics_list = []
            while self._test_schedule.update():
                self.train_mode(False)
                episode_metrics = self.run_episode()
                episode_metrics_list.append(episode_metrics)

            if len(episode_metrics_list) != 0:
                mean_episode_metrics = self.create_episode_metrics()
                for agent in self._agents:
                    mean_episode_metrics[agent.id]["reward"] = mean([ep_metrics[agent.id]["reward"] for ep_metrics in episode_metrics_list])
                    mean_episode_metrics[agent.id]["disc_reward"] = mean([ep_metrics[agent.id]["disc_reward"] for ep_metrics in episode_metrics_list])
                    mean_episode_metrics[agent.id]["episode_length"] = mean([ep_metrics[agent.id]["episode_length"] for ep_metrics in episode_metrics_list])
                mean_episode_metrics["full_episode_length"] = mean([ep_metrics["full_episode_length"] for ep_metrics in episode_metrics_list])
                mean_episode_metrics["env_steps"] = self._train_step_schedule._steps
                self._logger.update_step("test_episodes")
                self._logger.log_metrics(mean_episode_metrics.get_flat_dict(), "test_episodes")

            # Save experiment state
            if self._experiment_manager.update_step():
                self._experiment_manager.save()

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

    # Set up environment
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
            agent_config["kwargs"]["env_info"] = env_spec.env_info
            agent_config["kwargs"]["logger"] = logger

            if "replay_buffer" in agent_config["kwargs"]:
                replay_args = agent_config["kwargs"]["replay_buffer"]["kwargs"]
                replay_args["observation_shape"] = env_spec.obs_dim[idx]
            agent, full_agent_config = agent_lib.get_agent(agent_config, f"agent.{idx}")
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
