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

Each of these components can be independently used by with our 
:py:class:`~hive.agents.rainbow.RainbowDQNAgent` class. To use Prioritized Replay,
you must pass a :py:class:`~hive.replays.prioritized_replay.PrioritizedReplayBuffer`
to the :py:obj:`replay_buffer` parameter of 
:py:class:`~hive.agents.rainbow.RainbowDQNAgent`. The details for how to use the other
components of rainbow are found in the API documentation of 
:py:class:`~hive.agents.rainbow.RainbowDQNAgent`.


Other Customizations
-------------------------------------
There are many other customizations that you can do to
:py:class:`~hive.agents.rainbow.RainbowDQNAgent` or 
:py:class:`~hive.agents.dqn.DQNAgent` by simply overriding some functions. 


