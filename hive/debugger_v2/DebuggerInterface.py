from pathlib import Path
from hive.utils.registry import Registrable
from hive.debugger_v2.utils import settings


class DebuggerInterface(Registrable):
    def __init__(self):
        self.main_msgs = settings.load_messages()
        self.config = None
        self.check_type = None
        self.check_period = None
        self.iter_num = None

    @classmethod
    def type_name(cls):
        return "debugger"