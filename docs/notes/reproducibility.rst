Reproducibility
================

Achieving reproducibility in deep RL is difficult. Even when the random seed
is fixed, libraries such as PyTorch use algorithms and implementations that
are nondeterministic. PyTorch has several options that allow the user to
turn off some aspects of this nondeterminism, **but behavior is still usually
only replicable if the runs are executed on the same hardware**. 

We provide a global seeding class :class:`~hive.utils.utils.Seeder` 
that allows the user to set a global seed for all packages currently 
used by the framework (NumPy, PyTorch, and Python's random package). It also
sets the PyTorch options to turn off nondeterminism. When using this seeding
functionality, before starting a run, you must set the environment variable
``CUBLAS_WORKSPACE_CONFIG`` to either ``":16:8"`` (limits performance) or
``":4096:8"`` (uses slightly more memory). See 
`this page <https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility>`_
for more details.

The :class:`~hive.utils.utils.Seeder`  class also provides a function 
:meth:`~hive.utils.utils.Seeder.get_new_seed` that provides a new
random seed each time it is called, which is useful when in multi-agent setups where
you want each agent to be seeded differently.
