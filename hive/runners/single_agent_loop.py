import argparse
import numpy as np
import torch
from hive import agents, envs, replays
from hive.agents import qnets
from hive.utils import logging, schedule, utils, experiment


def run_single_agent_training(
    environment, agent, training_schedule, testing_schedule, logger, experiment_manager,
):
    agent.train()
    episode_metrics = {
        "cumulative_reward": 0,
        "episode_length": 0,
    }
    observation, _ = environment.reset()
    while training_schedule.update():
        done, observation = run_one_step(
            environment, agent, observation, episode_metrics
        )

        if done:
            if logger.update_step("train_episodes"):
                logger.log_metrics(episode_metrics, "train_episodes")
            episode_metrics = {
                "cumulative_reward": 0,
                "episode_length": 0,
            }
            while testing_schedule.update():
                # Run the testing loop for one episode
                agent.eval()
                done = False
                observation, _ = environment.reset()
                while not done:
                    done, observation = run_one_step(
                        environment, agent, observation, episode_metrics
                    )

                if logger.update_step("test_episodes"):
                    logger.log_metrics(episode_metrics, "test_episodes")

                # Reset the agent for training
                agent.train()
                episode_metrics = {
                    "cumulative_reward": 0,
                    "episode_length": 0,
                }
            observation, _ = environment.reset()
        if experiment_manager.update_step():
            experiment_manager.save()
    experiment_manager.save()


def run_one_step(environment, agent, observation, episode_metrics):
    action = agent.act(observation)
    next_observation, reward, done, turn, info = environment.step(action)
    agent.update(
        {
            "observation": observation,
            "action": action,
            "reward": reward,
            "next_observation": next_observation,
            "done": done,
            "turn": turn,
            "info": info,
        }
    )
    episode_metrics["cumulative_reward"] += reward
    episode_metrics["episode_length"] += 1
    return done, next_observation


def set_up_dqn_experiment(args):
    environment = envs.GymEnv(args.environment)
    env_spec = environment.env_spec
    external_logger = logging.get_logger(
        args.logger_type,
        project_name=args.project_name,
        run_name=args.run_name,
        timescales=["train_episodes", "test_episodes"],
        logger_schedules={
            "agent": schedule.PeriodicSchedule(False, True, args.agent_log_frequency),
        },
        offline=args.logger_offline,
    )
    chomp_logger = logging.get_logger(
        "chomp", timescales=["train_episodes", "test_episodes"],
    )
    logger = logging.CompositeLogger(
        timescales=["train_episodes", "test_episodes", "agent"],
        logger_list=[external_logger, chomp_logger],
    )
    agent = agents.DQNAgent(
        qnet=qnets.SimpleMLP(env_spec),
        env_spec=env_spec,
        optimizer=torch.optim.Adam,
        replay_buffer=replays.CircularReplayBuffer(
            np.random.default_rng(args.random_seed),
            size=args.replay_size,
            compress=args.replay_compress,
        ),
        discount_rate=args.discount_rate,
        grad_clip=args.grad_clip,
        target_net_soft_update=args.target_net_soft_update,
        target_net_update_fraction=args.target_net_update_fraction,
        target_net_update_schedule=schedule.PeriodicSchedule(
            False, True, args.target_net_update_period
        ),
        epsilon_schedule=schedule.LinearSchedule(
            args.epsilon_start, args.epsilon_end, args.epsilon_steps
        ),
        learn_schedule=schedule.SwitchSchedule(False, True, args.learn_start),
        seed=args.random_seed,
        batch_size=args.batch_size,
        device=args.device,
        logger=logger,
        log_frequency=args.agent_log_frequency,
    )
    training_schedule = schedule.SwitchSchedule(True, False, args.training_steps)
    testing_schedule = schedule.DoublePeriodicSchedule(
        False, True, args.test_frequency, args.num_test_episodes
    )
    saving_schedule = schedule.PeriodicSchedule(False, True, args.save_freq)
    saving_schedule.update()  # Don't save on first step
    config = utils.Chomp()
    config.add_from_dict(vars(args))
    config.training_schedule = training_schedule
    config.testing_schedule = testing_schedule
    config.saving_schedule = saving_schedule
    experiment_manager = experiment.Experiment(
        args.run_name, args.save_dir, saving_schedule
    )
    experiment_manager.register_experiment(
        config=config, agents=agent, logger=logger,
    )
    if args.resume:
        experiment_manager.resume()
        training_schedule = config.training_schedule
        testing_schedule = config.testing_schedule
        saving_schedule = config.saving_schedule

    return (
        environment,
        agent,
        training_schedule,
        testing_schedule,
        logger,
        experiment_manager,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", default="CartPole-v0")
    parser.add_argument("-p", "--project-name", default="Hive-v1")
    parser.add_argument("-n", "--run-name", default="test-run")
    parser.add_argument("-l", "--agent-log-frequency", type=int, default=50)
    parser.add_argument("-t", "--training-steps", type=int, default=1000000)
    parser.add_argument("-r", "--random-seed", type=int, default=42)
    parser.add_argument("--save-dir", default="./experiment")
    parser.add_argument("--save-freq", type=int, default=100000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--logger-offline", action="store_true")
    parser.add_argument("--logger-type", default="wandb", choices=["wandb", "null"])
    parser.add_argument("--test-frequency", type=int, default=10)
    parser.add_argument("--num-test-episodes", type=int, default=1)
    parser.add_argument("--replay-size", type=int, default=100000)
    parser.add_argument("--replay-compress", type=bool, default=False)
    parser.add_argument("--discount-rate", type=float, default=0.99)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--target-net-soft-update", type=bool, default=False)
    parser.add_argument("--target-net-update-fraction", type=float, default=0.05)
    parser.add_argument("--target-net-update-period", type=int, default=10000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.1)
    parser.add_argument("--epsilon-steps", type=int, default=1000000)
    parser.add_argument("--learn-start", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    (
        environment,
        agent,
        training_schedule,
        testing_schedule,
        logger,
        experiment_manager,
    ) = set_up_dqn_experiment(args)
    run_single_agent_training(
        environment,
        agent,
        training_schedule,
        testing_schedule,
        logger,
        experiment_manager,
    )

