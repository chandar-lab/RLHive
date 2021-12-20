Using the DQN/Rainbow Agents
===============================

The :py:class:`~hive.agents.dqn.DQNAgent` and :py:class:`~hive.agents.rainbow.RainbowDQNAgent`
are written to allow for easy extensions and adaptation to your applications. We outline a few different use cases here.

Using a different network architecture
--------------------------------------
Using different types of network architectures with
:py:class:`~hive.agents.dqn.DQNAgent` and :py:class:`~hive.agents.rainbow.RainbowDQNAgent`
is done using the ``representation_net`` parameter in the constructor. This network
should not include the final layer which computes the final Q-values. It
computes the representations that are fed into the layer which will compute the
final Q-values. This is because often the only difference between different variations
of the DQN algorithms is how the final Q-values are computed, with the rest of the architecture
not changing.

You can modify the architecture of the representation network from the config, or create 
a completely new architecture better suited to your needs. From the config, two different
types of network architectures are supported:

* :py:class:`~hive.agents.qnets.conv.ConvNetwork`: Networks with convolutional layers, followed by an MLP
* :py:class:`~hive.agents.qnets.mlp.MLPNetwork`: An MLP with only linear layers

See :ref:`this page <tutorials/configuration_tutorial:configuration>` 
for details on how to configure the network.

To use an architecture not supported by the above classes, simply write the Pytorch
module implementing the architecture, and register the class wrapped with 
:py:class:`~hive.agents.qnets.base.FunctionApproximator` wrapper. The only requirement is that this class should take
in the input dimension as the first positional argument:

.. code-block:: python
    
    import torch

    import hive
    from hive.agents.qnets import FunctionApproximator

    class CustomArchitecture(torch.nn.Module):
        def __init__(self, in_dim, hidden_units):
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(in_dim, hidden_units),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_units, hidden_units)
            )

        def forward(self, x):
            x = torch.flatten(x, start_dim=1)
            return self.network(x)
    
    hive.registry.register(
        'CustomArchitecture', 
        FunctionApproximator(CustomArchitecture), 
        FunctionApproximator
    )

Adding in different Rainbow components
--------------------------------------
The Rainbow architecture is composed of several different components, namely:

* Double Q-learning
* Prioritized Replay
* Dueling Networks
* Multi-step Learning
* Distributional RL
* Noisy Networks

Each of these components can be independently used with our 
:py:class:`~hive.agents.rainbow.RainbowDQNAgent` class. To use Prioritized Replay,
you must pass a :py:class:`~hive.replays.prioritized_replay.PrioritizedReplayBuffer`
to the ``replay_buffer`` parameter of 
:py:class:`~hive.agents.rainbow.RainbowDQNAgent`. The details for how to use the other
components of rainbow are found in the API documentation of 
:py:class:`~hive.agents.rainbow.RainbowDQNAgent`.


Custom Input Observations
-------------------------------------
The current implementations of :py:class:`~hive.agents.dqn.DQNAgent`
and :py:class:`~hive.agents.rainbow.RainbowDQNAgent` handle the standard case of 
observations being a single numpy array, and no extra inputs being necessary during
the update phase other than  ``action``, ``reward``, and ``done``. In the situation
where this is not the case, and you need to handle more complex inputs, you can do so
by overriding the methods of :py:class:`~hive.agents.dqn.DQNAgent`. Let's walk through
the example of :py:class:`~hive.agents.legal_moves_rainbow.LegalMovesRainbowAgent`. 
This agent takes in a list of legal moves on each turn and only selects from those.

.. code-block:: python

    class LegalMovesHead(torch.nn.Module):
        def __init__(self, base_network):
            super().__init__()
            self.base_network = base_network

        def forward(self, x, legal_moves):
            x = self.base_network(x)
            return x + legal_moves

        def dist(self, x, legal_moves):
            return self.base_network.dist(x)

    class LegalMovesRainbowAgent(RainbowDQNAgent):
        """A Rainbow agent which supports games with legal actions."""

        def create_q_networks(self, representation_net):
            """Creates the qnet and target qnet."""
            super().create_q_networks(representation_net)
            self._qnet = LegalMovesHead(self._qnet)
            self._target_qnet = LegalMovesHead(self._target_qnet)

This defines a wrapper around the Q-networks used by agent that takes an
encoding of the legal moves where illegal moves have value :math:`-\infty`
and legal moves have value :math:`0`. The wrapper then adds this encoding
to the values generated by the base Q-networks. Overriding 
:py:meth:`~hive.agents.dqn.DQNAgent.create_q_networks` allows you to modify the
base Q-networks by adding this wrapper.

.. code-block:: python

        def preprocess_update_batch(self, batch):
            for key in batch:
                batch[key] = torch.tensor(batch[key], device=self._device)
            return (
                (batch["observation"], batch["action_mask"]),
                (batch["next_observation"], batch["next_action_mask"]),
                batch,
            )

Now, since the Q-networks expect an extra parameter (the legal moves action mask),
we override the :py:meth:`~hive.agents.dqn.DQNAgent.preprocess_update_batch` method,
which takes a batch sampled from the replay buffer and defines the inputs that will
be used to compute the values of the current state and the next state during the update
step.

.. code-block:: python

        def preprocess_update_info(self, update_info):
            preprocessed_update_info = {
                "observation": update_info["observation"]["observation"],
                "action": update_info["action"],
                "reward": update_info["reward"],
                "done": update_info["done"],
                "action_mask": action_encoding(update_info["observation"]["action_mask"]),
            }
            if "agent_id" in update_info:
                preprocessed_update_info["agent_id"] = int(update_info["agent_id"])
            return preprocessed_update_info

We must also make sure that the action encoding for each transition is added to the
replay buffer in the first place. To do that, we override the 
:py:meth:`~hive.agents.dqn.DQNAgent.preprocess_update_info` method, which should return
a dictionary with keys and values corresponding to the items you wish to store into the
replay buffer. Note, these keys need to be specified when you create the replay buffer,
see :ref:`Replays <replays>` for more information.

.. code-block:: python

        @torch.no_grad()
        def act(self, observation):
            if self._training:
                if not self._learn_schedule.get_value():
                    epsilon = 1.0
                elif not self._use_eps_greedy:
                    epsilon = 0.0
                else:
                    epsilon = self._epsilon_schedule.update()
                if self._logger.update_step(self._timescale):
                    self._logger.log_scalar("epsilon", epsilon, self._timescale)
            else:
                epsilon = self._test_epsilon

            vectorized_observation = torch.tensor(
                np.expand_dims(observation["observation"], axis=0), device=self._device
            ).float()
            legal_moves_as_int = [
                i for i, x in enumerate(observation["action_mask"]) if x == 1
            ]
            encoded_legal_moves = torch.tensor(
                action_encoding(observation["action_mask"]), device=self._device
            ).float()
            qvals = self._qnet(vectorized_observation, encoded_legal_moves).cpu()

            if self._rng.random() < epsilon:
                action = np.random.choice(legal_moves_as_int).item()
            else:
                action = torch.argmax(qvals).item()

            return action


Finally, you also need to override the :py:meth:`~hive.agents.dqn.DQNAgent.act` method
to extract and use the extra information.
