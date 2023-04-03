import argparse
from pprint import pprint
from hive.runners.utils import load_config
from hive.runners import get_runner


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
    # import pdb 
    # pdb.set_trace()
    runner_fn, full_config = get_runner(config)
    runner = runner_fn()
    runner.register_config(full_config)
    if args.resume:
        runner.resume()
    runner.run_training()


if __name__ == "__main__":
    main()
