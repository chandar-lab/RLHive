import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.registration import register


class MABC(gym.Env):
    """
    Ref: Mehnaz Mannan thesis
        Stylized example of a communication system in which two devices transmit over a multiple access channel.

    """

    def __init__(self):
        """
        Number of devices = 2
        All other variables initialized below instead of using arguments as we use Open AI Gym Registered Environment
        Values used from Mehnaz Mannan's thesis
        """
        super(MABC, self).__init__()

        self.p = np.array(
            [0.6, 0.9]
        )  # Arrival (success) probability: vector of length 2 (p=0.3,0.6)-jalal's paper
        self.name = "MABC" + "_" + str(self.p[0]) + "_" + str(self.p[1])
        self.get_W = lambda: np.array(
            [np.random.binomial(1, self.p[n]) for n in range(2)]
        )  # Arrival Bernoulli processes

        self.prev_actions = np.zeros(2).astype(int)
        self.N_0 = np.zeros(2).astype(
            int
        )  # Initial buffer size. This is the initial "local state" {0,1}
        self.N = self.N_0  # Buffer size/Local state {0,1}

        self.S_0 = 0  # Initial channel state ∈ {0,1} = {Idle, Busy}
        self.S = self.S_0  # Channel state at t

        α_0 = 0.9
        α_1 = 0.9
        self.P = np.array(
            [[α_0, 1 - α_0], [1 - α_1, α_1]]
        )  # Transition matrix of the channel state Markov process

        self.c = 0.6
        self.r = 1

        self.no_obs = 2  # \mathfrak{C}
        self.num_channel_obs = 3
        self.channel_obs = self.S_0

        # 2 agents can send or not send
        self.action_space = gym.spaces.Tuple(
            tuple(gym.spaces.Discrete(2) for agent in range(2))
        )

        #1st bin - size 3 - 0,1,2 (idle,busy,no obs)
        #2nd bin - size 2 - 0,1 (local state is "no packet in buffer" or "packet in buffer")
        #3rd bin - size 2 - 0,1 (action of agent 0)
        #4th bin - size 2 - 0,1 (action of agent 1)
        #5th bin - size 2 - 0,1 (agent 0 or agent 1's obs)
        self.observation_space = gym.spaces.Tuple(
            tuple(gym.spaces.Discrete(48) for agent in range(2))
        )

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, actions):
        """
        prescription ∈ {0, 1, 2} = {only device 1 gets transmit opportunity, only device 2 gets transmit opportunity, both devices get transmit opportunity}
        """
        U = np.minimum(actions, self.N).astype(
            int
        )  # Device can transmit if it gets a transmit opportunity and it has a packet in the buffer
        W = self.get_W()
        # print("Buffer", self.N)
        # print("Channel state", self.S)
        # print("Actions", U)
        # print("Arrival", W)

        no_collision = U[0] ^ U[1]
        success = no_collision and (self.S == 0)

        # Buffer evolution
        self.N = np.minimum(self.N - U * success + W, 1)

        # Reward computation
        reward = (
            -np.sum(U) * self.c + success * self.r
        )  # Reward if only 1 transmits! ^: XOR operator in Python

        self.channel_obs = self.S if sum(U) > 0 else self.no_obs

        # Channel state evolution
        self.S = np.random.multinomial(1, self.P[self.S]).argmax()

        self.prev_actions = np.array(actions).astype(int)

        return self.gen_obs(), reward, False, {}

    def reset(self):
        """
        A function that resets the environment.
        Input:None
        Output:
        state:
        """
        self.N = self.N_0
        self.S = self.S_0
        self.channel_obs = self.S_0
        self.prev_actions = np.zeros(2).astype(int)

        return self.gen_obs()

    def gen_agent_obs(self, agent):
        # for agent
        #1st bin - size 3 - 0,1,2 (idle,busy,no obs)
        #2nd bin - size 2 - 0,1 (local state of agent is "no packet in buffer" or "packet in buffer")
        #3rd bin - size 2 - 0,1 (prev action of agent 0)
        #4th bin - size 2 - 0,1 (prev action of agent 1)
        #5th bin - size 2 - 0,1 (agent 0 or agent 1's obs)
        agent_obs = 24*agent + 12*self.prev_actions[1] + 6*self.prev_actions[0] + 3*self.N[agent] + self.channel_obs
        return np.uint8(agent_obs)

    def gen_obs(self):
        return [self.gen_agent_obs(agent) for agent in range(2)]
        