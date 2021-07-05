[![Python unit tests for Hive](https://github.com/chandar-lab/RLHive/actions/workflows/pull_request_ci.yml/badge.svg)](https://github.com/chandar-lab/RLHive/actions/workflows/pull_request_ci.yml) [![Black Linter](https://github.com/chandar-lab/RLHive/actions/workflows/linter.yml/badge.svg)](https://github.com/chandar-lab/RLHive/actions/workflows/linter.yml)

[**Installing**](#installing) | [**Configuring**](#creating-an-experiment) | [**Running**](#running) | [**Contributing**](#contributing)
# RLHive
RLHive is a library of reinforcement learning agents and all the other components needed to create a reinforcement learning experiment. We provide support for both single agent and multi agent environments. 
## Installing
To install the necessary dependencies for RLHive, simply run
```
pip install -r requirements.txt
```  
from the root directory of the repository. We strongly recommend that you do this inside of a conda or virtualenv environment, so that you don't get conflicting dependencies.   
## Creating an experiment
Experiments are specified using config yaml files. For examples, please see the [configs](configs/) directory. The general structure of each is as follows:
```
run_name: ...
# Other training loop arguments
environment:
    name: ... # Name of Environment class
    kwargs:
        # Arguments used to create environment
        ...
agents:
    - 
        name: ... # Name of Agent 1 class
        kwargs:
            # Agent 1 arguments
            ...
    -
        name: ... # Name of Agent 1 class
        kwargs:
            # Agent 2 arguments
            ...
    ...
loggers:
    - 
        name: ... # Name of Logger 1 class
        kwargs:
            # Logger 1 arguments
            ...
    -
        name: ... # Name of Logger 1 class
        kwargs:
            # Logger 2 arguments
            ...
    ...
```
All of the registered class names are located in the `__init__.py` files of the relevant folders.

The configurations for environment, agents, and loggers can also be put into separate files if needed. 
Certain arguments can be overriden from the command line as well. Specifically, arguments listed in the runner file, and any arguments of registered Hive objects.

## Hive Registry
The `Registrable` class denotes which types of objects can be registered in the Hive
Registry. These objects can be configured directly from the command line. To use this
functionality, simply subclass `Registrable`, and define a function `type_name()` that
returns a string denoting the class's type name. You only need to do this for base
classes. For example, the `Agent` class has a `type_name()` function that returns the
string `'agent'`, and all derived classes simply inherit that `type_name()`.

The Hive registry allows you to register different types
of (`Registrable`) classes and objects and generates wrapper constructors for those
classes in the form of `get_{type_name}`.

These wrapper constructors allow you
to specify/override arguments for object constructors directly from the
command line. These parameters are specified in dot notation. They also are able
to handle lists and dictionaries of Registrable objects.

For example, let's consider the following scenario:
Your agent class has an argument `arg1` which is annotated to be `List[Class1]`,
`Class1` is `Registrable`, and the `Class1` constructor takes an argument `arg2`.
In the passed yaml config, there are two different Class1 object configs listed.
the constructor will check to see if both `--agent.arg1.0.arg2` and
`--agent.arg1.1.arg2` have been passed to the command line. If so, that value
will override whatever was in the config.

The parameters passed in the command line will be parsed according to the type
annotation of the corresponding low level constructor. If it is not one of
`int`, `float`, `str`, or `bool`, it simply loads the string into python using a
yaml loader.

## Running
Once the config has been created, running an experiment is simple. For a single agent CartPole experiment, navigate to the root directory, and simply run:
```
python -m hive.runners.single_agent_loop -c configs/cartpole_dqn/config.yml
```

For multiagent experiments, such as independent DQNs with Marlgrid, navigate to the root directory and run:
```
python -m hive.runners.multi_agent_loop -c configs/marlgrid_ma2_9x9_dqn/config.yml
```

## Contributing
When contributing to RLHive, please follow these guidelines:

- Create a new branch for each feature you are adding. When you are done writing the feature, create a pull request to the dev branch. Each pull request must pass all the unit tests and be approved by two other people before being merged into dev.
- Run the [black](https://black.readthedocs.io/en/stable/editor_integration.html) formatter before committing code. That will ensure that we have a uniform code style for the repo. We will be adding linter style checks as a requirement for pull requests soon.
- Make sure you document your code. Any class or semi-complicated function you write should have a docstring.
- If you add new features, add unit tests to test the feature. To run unit tests locally, navigate to the root directory and run:
```
python -m pytest
```    
