.. _registration:

Registration
==================

Registering new Classes
-----------------------
To register a new class with the Registry, you need to make sure that it or one of its 
ancestors subclassed :py:class:`hive.utils.registry.Registrable` and provided a definition
for :py:meth:`hive.utils.registry.Registrable.type_name`. The return value of this function
is used to create the getter function for that type (``registry.get_{type_name}``). 

You can either register several different classes at once:

.. code-block:: python
    
    registry.register_all(
        Agent,
        {
            "DQNAgent": DQNAgent,
            "LegalMovesRainbowAgent": LegalMovesRainbowAgent,
            "RainbowDQNAgent": RainbowDQNAgent,
            "RandomAgent": RandomAgent,
        },
    )

or one at a time:

.. code-block:: python

    registry.register("DQNAgent", DQNAgent, Agent)
    registry.register("LegalMovesRainbowAgent", LegalMovesRainbowAgent, Agent)
    registry.register("RainbowDQNAgent", RainbowDQNAgent, Agent)
    registry.register("RandomAgent", RandomAgent, Agent)

After a class has been registered, you can use pass a config dictionary to the getter function
for that type to create the object.

Callables
----------
There are several cases where we want to parameterize some function or constructor partway, but
not pass the fully created object in as an argument. One example is optimizers. You might want
to pass a learning rate, but you cannot create the final optimizer object until you've created
the parameters you want to optimize. To deal with such cases, we provide a 
:py:class:`~hive.utils.registry.CallableType` class, which can be used to register and wrap any 
callable. For example, with optimizers, we have:

.. code-block:: python

    class OptimizerFn(CallableType):
        """A wrapper for callables that produce optimizer functions.

        These wrapped callables can be partially initialized through configuration
        files or command line arguments.
        """

        @classmethod
        def type_name(cls):
            """
            Returns:
                "optimizer_fn"
            """
            return "optimizer_fn"
    
    registry.register_all(
        OptimizerFn,
        {
            "Adadelta": OptimizerFn(optim.Adadelta),
            "Adagrad": OptimizerFn(optim.Adagrad),
            "Adam": OptimizerFn(optim.Adam),
            "Adamax": OptimizerFn(optim.Adamax),
            "AdamW": OptimizerFn(optim.AdamW),
            "ASGD": OptimizerFn(optim.ASGD),
            "LBFGS": OptimizerFn(optim.LBFGS),
            "RMSprop": OptimizerFn(optim.RMSprop),
            "RMSpropTF": OptimizerFn(RMSpropTF),
            "Rprop": OptimizerFn(optim.Rprop),
            "SGD": OptimizerFn(optim.SGD),
            "SparseAdam": OptimizerFn(optim.SparseAdam),
        },
    )

With this, we can now make use of the configurability of RLHive objects while still 
passing callables as arguments.