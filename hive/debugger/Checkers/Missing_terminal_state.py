import numpy as np
from hive.debugger.DebuggerInterface import DebuggerInterface
from hive.debugger.utils.metrics import almost_equal


class MissingTerminalStateCheck(DebuggerInterface):

    def __init__(self, check_period):
        super().__init__()
        self.check_type = "Missing Terminal State"
        self.check_period = check_period
        self.iter_num = -1

    def run(self, env):
        pass
