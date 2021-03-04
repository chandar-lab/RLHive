import argparse
import gym
import numpy as np
import torch
from hive import agents, envs, replays
from hive.agents import qnets
from hive.utils import logging, schedule


def run_single_agent_training(environment, agent, logger, training_schedule):
    observation = environment.reset()
    cum_reward = 0
    while training_schedule.update():
        action = agent.act(obs_0, training=True)
        next_observation, reward, done, info = environment.step(action)
        agent.update(
            {
                "observation": observation,
                "action": action,
                "reward": reward,
                "next_observation": next_observation,
                "done": done,
                "info": info,
            }
        )
        cum_reward += reward

        if done:
            logger.update_step()
            if logger.should_log():
                logger.log_scalar("cum_reward", cum_reward)
            cum_reward = 0
            observation = environment.reset()
        else:
            observation = next_observation


def set_up_dqn_experiment(args):
    # This is v1 hardcoded experiment. Should be initialized using command line arguments
    environment = gym.make(args.environment)
    # environment = gym.make("MountainCar-v0")
    env_spec = envs.EnvSpec(
        args.environment,
        environment.observation_space.shape[0],
        environment.action_space.n,
    )
    agent_logger = logging.WandbLogger(
        args.project_name,
        args.run_name,
        logger_schedule=schedule.PeriodicSchedule(
            False, True, args.agent_log_frequency
        ),
        logger_name="agent",
    )
    rng = np.random.default_rng(seed=args.random_seed)
    agent = agents.DQNAgent(
        qnet=qnets.SimpleMLP(env_spec),
        env_spec=env_spec,
        optimizer=torch.optim.Adam,
        replay_buffer=replays.CircularReplayBuffer(
            rng, size=args.replay_size, compress=args.replay_compress
        ),
        discount_rate=args.discount_rate,
        target_net_soft_update=args.target_net_soft_update,
        target_net_update_fraction=args.target_net_update_fraction,
        target_net_update_schedule=schedule.PeriodicSchedule(
            False, True, args.target_net_update_period
        ),
        epsilon_schedule=schedule.LinearSchedule(
            args.epsilon_start, args.epsilon_end, args.epsilon_steps
        ),
        learn_schedule=schedule.SwitchSchedule(False, True, args.learn_start),
        rng=rng,
        batch_size=args.batch_size,
        device=args.device,
        logger=agent_logger,
    )
    training_schedule = schedule.SwitchSchedule(True, False, args.training_steps)
    logger = logging.WandbLogger(
        args.project_name,
        args.run_name,
        logger_schedule=schedule.ConstantSchedule(True),
        logger_name="runner",
    )
    return environment, agent, training_schedule, logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", default="CartPole-v1")
    parser.add_argument("-p" "--project-name", default="Hive-v1")
    parser.add_argument("-t" "--run-name", default="test-run")
    parser.add_argument("-l" "--agent-log-frequency", type=int, default=50)
    parser.add_argument("-t" "--training-steps", type=int, default=1000000)
    parser.add_argument("-r" "--random-seed", type=int, default=42)
    parser.add_argument("--replay-size", type=int, default=100000)
    parser.add_argument("--replay-compress", type=bool, default=False)
    parser.add_argument("--discount-rate", type=float, default=0.99)
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
    environment, agent, training_schedule, logger = set_up_experiment(args)
    run_single_agent_training(environment, agent, logger, training_schedule)
