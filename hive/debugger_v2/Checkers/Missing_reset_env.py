import numpy as np
from hive.debugger_v2.DebuggerInterface import DebuggerInterface


class MissingResetStateCheck(DebuggerInterface):

    def __init__(self, check_period):
        super().__init__()
        self.check_type = "Missing Reset State"
        self.check_period = check_period
        self.iter_num = -1

    def run(self, env):
        pass
