Agent
==================

Agent API
----------
Interacting with the agent happens primarily through two functions: ``agent.act()`` and 
``agent.update()``. 
``agent.act()`` takes in an observation and returns an action, while ``agent.update()`` takes in a 
dictionary consisting of the information relevant to the most recent transition and updates the agent.


Creating an Agent
-----------------
Let's create a new tabular Q-learning agent with discount factor ``gamma`` and learning rate ``alpha``, 
for some environment with one hot observations.
We want this agent to have an epsilon greedy policy, with the exploration rate decaying over 
``explore_steps`` from ``1.0`` to some value ``final_epsilon``.
First, we define the constructor:

.. code-block:: python
    
    import numpy as np
    import os

    import hive
    from hive.agents.agent import Agent
    from hive.utils.schedule import LinearSchedule

    class TabularQLearningAgent(Agent):
        def __init__(self, obs_dim, act_dim, gamma, alpha, explore_steps, final_epsilon, id=0):
            super().__init__(obs_dim, act_dim, id=id)
            self._q_values = np.zeros(obs_dim, act_dim)
            self._gamma = gamma
            self._alpha = alpha
            self._act_dim = act_dim
            self._epsilon_schedule = LinearSchedule(1.0, final_epsilon, explore_steps)

In this constructor, we created a numpy array to keep track of the Q-values for every
state-action pair, and a linear decay schedule for the epsilon exploration rate. Next,
let's create the act function:

.. code-block:: python
    
        def act(self, observation):
            # Return a random action if exploring
            if np.random.rand() < self._epsilon_schedule.update():
                return np.random.randint(self._act_dim)
            else:
                state = np.argmax(observation) # Convert from one-hot

                # Break ties randomly between all actions with max values
                max_value = np.amax(self._q_values[state])
                best_actions = np.where(self._q_values[state] == max_value)[0]
                return np.random.choice(best_actions)

Now, we write our update function, which updates the state of our agent:

.. code-block:: python
    
        def update(self, update_info):
            state = np.argmax(update_info["observation"])
            next_state = np.argmax(update_info["next_observation"])

            self._q_values[state, update_info["action"]] += self._alpha * (
                update_info["reward"]
                + self._gamma * np.amax(self._q_values[next_state])
                - self._q_values[state, action]
            )

Now, we can directly use this environment with the single agent or multi-agent runners.
Note ``act`` and ``update`` are framework agnostic, so you could implement it with any
(deep) learning framework, although most of our implemented agents are written in PyTorch.

If we write a save and load function for this agent, we can also take advantage of checkpointing
and resuming in the runner: 

.. code-block:: python
    
        def save(self, dname):
            np.save(os.path.join(dname, "qvalues.npy"), self._q_values)
            pickle.dump({"schedule": self._epsilon_schedule}, open("state.p", "wb"))

        def load(self, dname):
            self._q_values = np.load(os.path.join(dname, "qvalues.npy"))
            self._epsilon_schedule = pickle.load(open("state.p", "rb"))["schedule"]

Finally, we :ref:`register <registration>` our agent class, so that it can be found when setting up experiments
through the yaml config files and command line.

.. code-block:: python
    
    hive.registry.register('TabularQLearningAgent', TabularQLearningAgent, Agent)