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
