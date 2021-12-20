[![Python unit tests for Hive](https://github.com/chandar-lab/RLHive/actions/workflows/pull_request_ci.yml/badge.svg)](https://github.com/chandar-lab/RLHive/actions/workflows/pull_request_ci.yml) [![Black Linter](https://github.com/chandar-lab/RLHive/actions/workflows/linter.yml/badge.svg)](https://github.com/chandar-lab/RLHive/actions/workflows/linter.yml)

[**Installing**](#installing) | [**Tutorials**](#tutorials) | [**Contributing**](#contributing)

[![te](docs/hive.svg)](docs/hive.svg) 
# RLHive
RLHive is a framework designed to facilitate research in reinforcement learning. It provides the components necessary to run a full RL experiment, for both single agent and multi agent environments. It is designed to be readable and easily extensible, to allow users to quickly run and experiment with their own ideas.

The full documentation and tutorials are available at https://rlhive.readthedocs.io/.
## Installing
RLHive is available through pip! For the basic RLHive package, simply run 
``pip install rlhive``.

You can also install dependencies necessary for the environments that
RLHive comes with by running ``pip install rlhive[<env_names>]`` where 
``<env_names>`` is a comma separated list made up of the following: 
- atari
- gym_minigrid
- pettingzoo

In addition to these environments, Minatar and Marlgrid are also supported, but
need to be installed separately. 

To install Minatar, run
``pip install MinAtar@git+https://github.com/kenjyoung/MinAtar.git@8b39a18a60248ede15ce70142b557f3897c4e1eb``

To install Marlgrid, run
``pip install marlgrid@https://github.com/kandouss/marlgrid/archive/refs/heads/master.zip``

## Tutorials
- [Creating new agents](https://rlhive.readthedocs.io/en/stable/tutorials/agent_tutorial.html)
- [Using DQN/Rainbow Agents](https://rlhive.readthedocs.io/en/stable/tutorials/dqn_tutorial.html)
- [Using Environments/Creating new Environments](https://rlhive.readthedocs.io/en/stable/tutorials/env_tutorial.html)
- [Configuring your experiments through YAML files and command line](https://rlhive.readthedocs.io/en/stable/tutorials/configuration_tutorial.html)
- [Loggers and Scheduling](https://rlhive.readthedocs.io/en/stable/tutorials/logging_tutorial.html)
- [Registering Custom RLHive Objects](https://rlhive.readthedocs.io/en/stable/tutorials/registration_tutorial.html)
- [Using Replay Buffers](https://rlhive.readthedocs.io/en/stable/tutorials/replay_tutorial.html)
- [Single/Multi-Agent Runners](https://rlhive.readthedocs.io/en/stable/tutorials/runner_tutorial.html)


## Contributing
We'd love for you to contribute your own work to RLHive. Before doing so, please read our 
[contributing guide](https://rlhive.readthedocs.io/en/stable/contributing.html).