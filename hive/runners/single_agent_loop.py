import argparse

from hive.runners.multi_agent_loop import set_up_experiment, load_config


def set_up_single_agent_experiment(config):
    """Returns a runner object for a single agent experiment."""
    config["environment"]["kwargs"]["num_players"] = 1
    return set_up_experiment(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="./config.yml")
    parser.add_argument("-a", "--agent-config")
    parser.add_argument("-e", "--env-config")
    parser.add_argument("-l", "--logger-config")
    args = parser.parse_args()
    config = load_config(args)
    runner = set_up_single_agent_experiment(config)
    runner.run_training()
