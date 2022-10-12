import argparse
from pprint import pprint
from hive.runners.utils import load_config
from hive.runners import get_runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    parser.add_argument("-p", "--preset-config")
    parser.add_argument("-a", "--agent-config")
    parser.add_argument("-e", "--env-config")
    parser.add_argument("-l", "--logger-config")
    parser.add_argument("-r", "--resume")
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
    runner_fn, full_config = get_runner(config)
    runner = runner_fn()
    runner.register_config(full_config)
    if args.resume:
        runner.resume()
    runner.run_training()


if __name__ == "__main__":
    main()
