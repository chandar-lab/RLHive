import logging
import os

logging.basicConfig(
    format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s",
    level=logging.INFO,
)

from hive import agents, envs, replays, runners, utils
from hive.utils.registry import Registrable, registry

__version__ = "1.0.1"
