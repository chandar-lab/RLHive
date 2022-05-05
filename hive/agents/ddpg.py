from hive.agents.td3 import TD3
from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.utils import InitializationFn
from hive.replays import BaseReplayBuffer
from hive.utils.loggers import Logger
from hive.utils.utils import LossFn, OptimizerFn


class DDPG(TD3):
    def __init__(
        self,
        observation_space,
        action_space,
        representation_net: FunctionApproximator = None,
        actor_net: FunctionApproximator = None,
        critic_net: FunctionApproximator = None,
        init_fn: InitializationFn = None,
        actor_optimizer_fn: OptimizerFn = None,
        critic_optimizer_fn: OptimizerFn = None,
        critic_loss_fn: LossFn = None,
        stack_size: int = 1,
        replay_buffer: BaseReplayBuffer = None,
        discount_rate: float = 0.99,
        n_step: int = 1,
        grad_clip: float = None,
        reward_clip: float = None,
        soft_update_fraction: float = 0.005,
        batch_size: int = 64,
        logger: Logger = None,
        log_frequency: int = 100,
        update_frequency: int = 1,
        action_noise: float = 0,
        min_replay_history: int = 1000,
        device="cpu",
        id=0,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            representation_net=representation_net,
            actor_net=actor_net,
            critic_net=critic_net,
            init_fn=init_fn,
            actor_optimizer_fn=actor_optimizer_fn,
            critic_optimizer_fn=critic_optimizer_fn,
            critic_loss_fn=critic_loss_fn,
            n_critics=1,
            stack_size=stack_size,
            replay_buffer=replay_buffer,
            discount_rate=discount_rate,
            n_step=n_step,
            grad_clip=grad_clip,
            reward_clip=reward_clip,
            soft_update_fraction=soft_update_fraction,
            batch_size=batch_size,
            logger=logger,
            log_frequency=log_frequency,
            update_frequency=update_frequency,
            policy_update_frequency=1,
            action_noise=action_noise,
            target_noise=0.0,
            target_noise_clip=0.0,
            min_replay_history=min_replay_history,
            device=device,
            id=id,
        )
