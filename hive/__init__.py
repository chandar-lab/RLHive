import os

from hive import agents, envs, replays, runners, utils
from hive.utils.registry import Registrable, registry

with open(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "version.txt"))
) as f:
    __version__ = f.read().strip()
