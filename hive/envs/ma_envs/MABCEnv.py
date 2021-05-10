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
        '''
        Number of devices = 2
        All other variables initialized below instead of using arguments as we use Open AI Gym Registered Environment
        Values used from Mehnaz Mannan's thesis
        '''
        super(MABC, self).__init__()
        
        self.p = np.array([0.6, 0.9]) #Arrival (success) probability: vector of length 2 (p=0.3,0.6)-jalal's paper
        self.name = "MABC" + '_' + str(self.p[0]) + '_' + str(self.p[1])
        self.get_W = lambda : np.array([np.random.binomial(1, self.p[n]) for n in range(2)]) #Arrival Bernoulli processes

        self.N_0 = np.zeros(2).astype(int) #Initial buffer size. This is the initial "local state" {0,1}
        self.N = self.N_0 #Buffer size/Local state {0,1}

        self.S_0 = 0   #Initial channel state ∈ {0,1} = {Idle, Busy}
        self.S = self.S_0 #Channel state at t

        α_0 = 0.9
        α_1 = 0.9
        self.P = np.array([[α_0, 1 - α_0],[ 1- α_1, α_1]]) #Transition matrix of the channel state Markov process

        self.c = 0.6
        self.r = 1

        self.no_obs = 2 #\mathfrak{C}
        self.num_channel_obs = 3
        self.channel_obs = self.S_0
        
        self.action_space = gym.spaces.Tuple(tuple(gym.spaces.Discrete(2) for agent in range(2))) #2 agents can send or not send
        self.observation_space = gym.spaces.Tuple(tuple(gym.spaces.Discrete(6) for agent in range(2))) # obs: (receive packet or not) x (channel state busy or idle or unknown)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, actions):
        '''
        prescription ∈ {0, 1, 2} = {only device 1 gets transmit opportunity, only device 2 gets transmit opportunity, both devices get transmit opportunity}
        '''
        U = np.minimum(actions, self.N).astype(int)  #Device can transmit if it gets a transmit opportunity and it has a packet in the buffer
        W = self.get_W()
        # print("Buffer", self.N)
        # print("Channel state", self.S)
        # print("Actions", U)
        # print("Arrival", W)

        no_collision = U[0]^U[1]
        success = no_collision and (self.S == 0)
        
        #Buffer evolution
        self.N = np.minimum(self.N - U*success + W, 1)
        
        #Reward computation
        reward = -np.sum(U)*self.c + success*self.r #Reward if only 1 transmits! ^: XOR operator in Python

        self.channel_obs = self.S if sum(U) > 0 else self.no_obs

        #Channel state evolution
        self.S = np.random.multinomial(1, self.P[self.S]).argmax()

        return self.gen_obs(), reward, False, {}

    def reset(self):
        '''
        A function that resets the environment.
        Input:None
        Output:
        state:
        '''
        self.N = self.N_0
        self.S = self.S_0
        self.channel_obs = self.S_0

        return self.gen_obs()

    def gen_agent_obs(self, agent):
        #for agent
        #if local state x=0, obs is 0,1,2 (idle,busy,no obs)
        #if local state x=1, obs is 3,4,5 (idle,busy,no obs)
        if self.N[agent] == 0:
            agent_obs = self.channel_obs
        elif self.N[agent] == 1:
            agent_obs = 3 + self.channel_obs
        return agent_obs

    def gen_obs(self):
        return [self.gen_agent_obs(agent) for agent in range(2)]

# register(
#     id='MABC-v0', 
#     entry_point='gym.envs.marl.MABCEnv:MABC', 
#     max_episode_steps=300, 
# )