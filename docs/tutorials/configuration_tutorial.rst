.. _configuration:

Configuration
===============
RLHive was written to allow for fast configuration and iteration on configuration.
The base configuration for all parameters for the experiment is done through YAML
files. The majority of these parameters can be overriden through the command line.

Any object :ref:`registered <registration>` with RLHive can be configured directly
through YAML files or through the command line, without having to add any extra
argument parsers. 


.. _yaml-config:

YAML files
------------
Let's do a basic example of configuring an agent through YAML files. Specifically,
let's create a :py:class:`~hive.agents.dqn_agent.DQNAgent`.

.. code-block:: yaml

    agent:
      name: DQNAgent
      kwargs:
        representation_net:
          name: MLPNetwork
          kwargs:
            hidden_units: [256, 256]
        discount_rate: .9
        replay_buffer:
          name: CircularReplayBuffer
        reward_clip: 1.0

In this example, :py:class:`~hive.agents.dqn_agent.DQNAgent` , 
:py:class:`~hive.agents.qnets.mlp.MLPNetwork` , and
:py:class:`~hive.replays.circular_replay.CircularReplayBuffer` are all classes
registered with RLHive. Thus, we can do this configuration directly. When the
``registry`` getter function for agents 
:py:meth:`~hive.utils.registry.Registry.get_agent`, is then called with this config
dictionary (with the missing required arguments such as ``obs_dim`` and ``act_dim``,
filled in), it will build all the inner RLHive objects automatically.
This works by using the type annotations on the constructors of the objects, so
to recursively create the internal objects, those arguments need to be annotated
correctly.


.. _override-config:

Overriding from command lines
--------------------------------
When using the ``registry`` getter functions, RLHive automatically checks any command 
line arguments passed to see if they match/override any default or yaml configured 
arguments. With ``getter`` function you provide a config and a prefix. That prefix
is added, prepended to any argument names when searching the command line. For example,
with the above config, if it were loaded and the 
:py:meth:`~hive.utils.registry.Registry.get_agent` method was called as follows:

.. code-block:: python

    agent = get_agent(config['agent'], 'ag')

Then, to override the discount_rate, you could pass the following argument to your
python script: ``--ag.discount_rate .95``. This can go arbitrarily deep into registered
RLHive class. For example, if you wanted to change the capacity of the replay buffer,
you could pass ``--ag.replay_buffer.capacity 100000``.

If the type annotation of the argument ``arg`` is ``List[C]`` where C is a registered
RLHive class, then you can override the argument of an individual object, ``foo``,
configured through YAML by passing ``--arg.0.foo <value>``.

Note as of this version, you must have configured the object in the YAML file in order
to override its parameters through the command line.