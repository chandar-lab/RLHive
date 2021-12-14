Quickstart
===========

Installation
^^^^^^^^^^^^^
Hive is available through pip! For the basic hive package, simply run 
``pip install rlhive``.

You can also install dependencies necessary for the environments that
Hive comes with by running ``pip install rlhive[<env_names>]`` where 
``<env_names>`` is a comma separated list made up of the following: 

* atari
* minatar
* minigrid
* marlgrid
* pettingzoo

Running an experiment
^^^^^^^^^^^^^^^^^^^^^
To get started with running an experiment, simply load a config, use it to create a 
runner, and run! This following code block trains a DQN on the Atari game Asterix
according to the specification of 
`Dopamine <https://github.com/google/dopamine/blob/master/dopamine/agents/dqn/configs/dqn.gin/>`_.

.. code-block:: python
    
    import hive
    from hive.runners.utils import load_config
    from hive.runners.single_agent_loop import set_up_experiment
    
    config = load_config(preset_config='atari/dqn.yml')
    runner = set_up_experiment(config)
    runner.run_training()

You can also run multi-agent experiments in a similar fashion. To replicate 
`DeepMind's Rainbow experiments on Hanabi 
<https://github.com/deepmind/hanabi-learning-environment/blob/master/hanabi_learning_environment/agents/rainbow/configs/hanabi_rainbow.gin>`_
, simply run: 

.. code-block:: python
    
    import hive
    from hive.runners.utils import load_config
    from hive.runners.multi_agent_loop import set_up_experiment
    
    config = load_config(preset_config='hanabi/self_play_rainbow.yml')
    runner = set_up_experiment(config)
    runner.run_training()


Hive comes with a number of preset configs that have been tested and shown to 
reproduce the results of their orignal papers. To run your own experiments,
simply :ref:`create your own config files <yaml-config>` or 
:ref:`override the parameters of the preset configs <override-config>`
from the command line.
