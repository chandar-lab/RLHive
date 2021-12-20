Loggers
=============


Using a Logger
--------------
RLHive currently provides 3 types of loggers:

* :py:class:`~hive.utils.loggers.ChompLogger`: Logs metrics to a dictionary like
  object that can be saved.
* :py:class:`~hive.utils.loggers.WandbLogger`: Logs metrics to WandB.
* :py:class:`~hive.utils.loggers.NullLogger`: Does not log metrics.

All of these loggers are :py:class:`~hive.utils.loggers.ScheduledLoggers`. 
When creating these loggers, you provide the different timescales that you want to
track with the logger. Timescales could correspond to any loop variable, such as
``"train_step"``, ``"agent_step"``, ``"test_iteration"``, etc.

With :py:class:`~hive.utils.loggers.ScheduledLoggers`, each
timescale is associated with a :py:class:`~hive.utils.schedule.Schedule` object. 
By updating these schedules in some loop, you can control the logging frequency
for that timescale. The loggers also keep track of how many times that timescale
was updated, and those values are logged alongside any metric you log (i.e. if 
``"timescale_1"`` was updated 7 times, then the logger will log an additional
key value pair of ``"timescale_1": 7`` with each metric. This allows
you to see the trends of any metric across any timescale. 

Timescales can be registered when creating the logger, or later on.

For an example:

.. code-block:: python

    from hive.utils.loggers import ChompLogger, CompositeLogger, WandbLogger
    from hive.utils.schedule import ConstantSchedule, PeriodicSchedule

    logger = ChompLogger(
        ['train_step', 'test_step'], # Timescales to track
        [ # How often to log train_step and test_step
            PeriodicSchedule(False, True, 10), # train_step is logged once every 10 times
            ConstantSchedule(True) # test_step is always logged
        ] 
    )

    # You can register timescales after logger creation as well. This particular timescale
    # is never scheduled to log.
    logger.register_timescale('dummy_timescale', ConstantSchedule(False))

    for _ in range(total_training_time):
        metrics_to_log = run_training_step()
        if logger.update_step('train_step'): # Evaluates to True once every 10 times it's hit.
            logger.log_metrics(metrics_to_log, 'training_metrics')


        other_metric = do_something_else()
        if logger.should_log('train_step'): # Checks schedule for this timescale without updating
            logger.log_scalar('other_metric', other_metric, 'training_metrics')
        

        if should_run_testing():
            test_metrics = run_testing()
            if logger.update_step('tets_step'): # Evaluates to True every time
                logger.log_metrics(test_metrics, 'testing_metrics')

Note, the schedules associated with each timescale are not hard constraints on when
you can log. They are merely a convenience to help you keep track of when to log.


Composite Logger
-----------------
If you want to log to multiple sources, without the hassle of keeping track of multiple
loggers, you can create a :py:class:`~hive.utils.loggers.CompositeLogger` object
initialized with each of the individual loggers that you want to use. For example:

.. code-block:: python

    from hive.utils.loggers import ChompLogger, CompositeLogger, WandbLogger

    logger = CompositeLogger([
        ChompLogger('train'),
        WandbLogger('train')
    ])

You can now use this logger as above, and it will log to both 
:py:class:`~hive.utils.loggers.ChompLogger` and 
:py:class:`~hive.utils.loggers.WandbLogger`.
