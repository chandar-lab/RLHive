Quickstart
===========

.. _installation:

Installation
^^^^^^^^^^^^^
RLHive is available through pip! For the basic RLHive package, simply run 
``pip install rlhive``.

You can also install dependencies necessary for the environments that
RLHive comes with by running ``pip install rlhive[<env_names>]`` where 
``<env_names>`` is a comma separated list made up of the following: 

* atari
* gym_minigrid
* pettingzoo

In addition to these environments, Minatar and Marlgrid are also supported, but
need to be installed separately. 

To install Minatar, run
``pip install MinAtar@git+https://github.com/kenjyoung/MinAtar.git@8b39a18a60248ede15ce70142b557f3897c4e1eb``

To install Marlgrid, run
``pip install marlgrid@https://github.com/kandouss/marlgrid/archive/refs/heads/master.zip``


Running an experiment
^^^^^^^^^^^^^^^^^^^^^
There are several ways to run an experiment with RLHive. If you want to just run a
preset config, you can directly run your experiment from the command line, with a config
file path relative to the
`hive/configs <https://github.com/chandar-lab/RLHive/hive/configs>`_ folder. These
examples run a DQN on the Atari game Asterix according to the
`Dopamine 
<https://github.com/google/dopamine/blob/master/dopamine/agents/dqn/configs/dqn.gin/>`_
configuration and a simplified Rainbow agent for Hanabi trained using self-play
according to the `DeepMind's 
<https://github.com/deepmind/hanabi-learning-environment/blob/master/hanabi_learning_environment/agents/rainbow/configs/hanabi_rainbow.gin>`_
configuration

.. code-block:: bash

    python -m hive_single_agent_loop -p atari/dqn.yml
    python -m hive_multi_agent_loop -p hanabi/rainbow.yml

If you want to run an experiment with components that are all available in RLHive,
but not presets, you can create your own config file, and run that instead! Make
sure you look at the examples 
`here <https://github.com/chandar-lab/RLHive/hive/configs>`_ and the tutorial
:ref:`here <yaml-config>` to properly format it:

.. code-block:: bash

    python -m hive_single_agent_loop -c <config-file>
    python -m hive_multi_agent_loop -c <config-file>

Finally, if instead you want to use your own custom custom components you can
simply register it with RLHive and run your config normally: 

.. code-block:: python
    
    import hive
    from hive.runners.utils import load_config
    from hive.runners.single_agent_loop import set_up_experiment
    
    class CustomAgent(hive.agents.Agent):
        # Definition of Agent
        pass
        
    hive.registry.register('CustomAgent', CustomAgent, CustomAgent)

    # Either load your custom full config file with that includes CustomAgent
    config = load_config(config='custom_full_config.yml')
    runner = set_up_experiment(config)
    runner.run_training()

    # Or load a preset config and just replace the agent config
    config = load_config(preset_config='atari/dqn.yml', agent_config='custom_agent_config.yml')
    runner = set_up_experiment(config)
    runner.run_training()

