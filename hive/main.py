import argparse
import logging
from pprint import pprint

from hive.runners import Runner
from hive.utils.registry import registry
from hive.utils.runner_utils import load_config

logging.basicConfig(
    format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s",
    level=logging.INFO,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to base config file.")
    parser.add_argument(
        "-p",
        "--preset-config",
        help="Path to preset base config in the RLHive repository. These are relative "
        "to the hive/configs/ folder in the repository. For example, the Atari DQN "
        "config would be atari/dqn.yml.",
    )
    parser.add_argument(
        "-a",
        "--agent-config",
        help="Path to the agent config. Overrides settings in base config.",
    )
    parser.add_argument(
        "-e",
        "--env-config",
        help="Path to environment configuration file. Overrides settings in base "
        "config.",
    )
    parser.add_argument(
        "-l",
        "--logger-config",
        help="Path to logger configuration file. Overrides settings in base config.",
    )
    parser.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="Whether to resume the experiment from given experiment directory",
    )
    args, config_unused_arguments = parser.parse_known_args()
    if args.config is None and args.preset_config is None:
        raise ValueError("Config needs to be provided")
    config = load_config(
        args.config,
        args.preset_config,
        args.agent_config,
        args.env_config,
        args.logger_config,
    )
    runner_fn, full_config, unused_args = registry.get(config, Runner)
    unused_args = set(unused_args) & set(config_unused_arguments)
    if len(unused_args) > 0:
        logging.warning(
            "The following command line arguments were not used in the experiment configuration: "
            f"{unused_args}"
        )
    runner = runner_fn()
    runner.register_config(full_config)
    if args.resume:
        runner.resume()
    runner.run_training()


if __name__ == "__main__":
    main()
