Runners
==================
We provide two different :py:class:`~hive.runners.base.Runner` classes: 
:py:class:`~hive.runners.single_agent_loop.SingleAgentRunner` and
:py:class:`~hive.runners.multi_agent_loop.MultiAgentRunner`. The setup
for both Runner classes can be viewed in their respective files with the
:py:meth:`set_up_experiment` functions. 
The :py:meth:`~hive.utils.registry.get_parsed_args` function can be used
to get any arguments from the command line that are not part of the signatures
of already registered RLHive class constructors. 


Metrics and TransitionInfo
---------------------------
The :py:class:`~hive.runners.utils.Metrics` class can be used to keep track
of metrics for single/multiple agents across an episode.

.. code-block:: python

    # Create the Metrics object. The first set of metrics is individual to
    # each agent, the second is common for all agents. The metrics can
    # be initialized either with a value or with callable with no arguments
    metrics = Metrics(
        [agent1, agent2],
        [("reward", 0), ("episode_traj", lambda: [])],
        [("full_episode_length", 0)],
    )

    # Add metrics
    metrics[agent1.id]["reward"] += 1
    metrics[agent2.id]["episode_traj"].append(0)
    metrics["full_episode_length"] += 1

    # Convert to flat dictionary for easy logging. Adds agent id's as prefixes
    # for agent_specific metrics
    flat_metrics = metrics.get_flat_dict()

    # Reinitialize/reset all metrics
    metrics.reset_metrics()


The :py:class:`~hive.runners.utils.TransitionInfo` class can be used to keep track
of the information needed by the agent to construct it's next state for acting or
next transition for updating. It also handles state stacking and padding.

.. code-block:: python

    transition_info = TransitionInfo([agent1, agent2], stack_size)
    
    # Set the start flag for agent1.
    transition_info.start(agent1)

    # Get stacked observation for agent1. If not enough observations have been
    # recorded, it will pad with 0s
    stacked_observation = transition_info.get_stacked_state(
        agent1, observation
    )

    # Record update information about the agent
    transition_info.record_info(agent, info)

    # Get the update information for the agent, with done set to the value passed
    info = transition_info.get_info(agent, done=done) 
    