import numpy as np
from hive.debugger.DebuggerInterface import DebuggerInterface
from hive.debugger.utils.metrics import almost_equal


class ObservationsCheck(DebuggerInterface):

    def __init__(self, check_period):
        super().__init__()
        self.check_type = "Observation"
        self.check_period = check_period
        self.iter_num = -1

    def run(self, observations):
        error_msg = list()
        self.iter_num += 1
        mas = np.max(observations)
        mis = np.min(observations)
        avgs = np.mean(observations)
        stds = np.std(observations)

        if stds == 0.0:
            error_msg.append(self.main_msgs['features_constant'])
        elif any([almost_equal(mas, data_max) for data_max in self.config["Data"]["normalized_data_maxs"]]) and \
                any([almost_equal(mis, data_min) for data_min in self.config["Data"]["normalized_data_mins"]]):
            return
        elif not (almost_equal(stds, 1.0) and almost_equal(avgs, 0.0)):
            error_msg.append(self.main_msgs['features_unnormalized'])
