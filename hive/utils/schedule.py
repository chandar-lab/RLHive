import abc


class Schedule(abc.ABC):
    @abc.abstractmethod
    def get_value():
        """Returns the current value of the variable we are tracking"""
        pass

    @abc.abstractmethod
    def update():
        """Update the value of the variable we are tracking and return the updated value."""
        pass


class LinearSchedule(Schedule):
    """Defines a linear schedule between two values over some number of steps.
    
    If updated more than the defined number of steps, the schedule stays at the
    end value.
    """

    def __init__(self, init_value, end_value, steps):
        """
        Args:
            init_value (Union[int, float]): starting value for schedule. 
            end_value (Union[int, float]): end value for schedule.
            steps (int): Number of steps for schedule. Should be positive.
        """
        steps = max(int(steps), 1)
        self._delta = (end_value - init_value) / steps
        self._end_value = end_value
        self._value = init_value + self._delta

    def get_value(self):
        return self._value

    def update(self):
        if self._value == self._end_value:
            return self._value

        self._value += self._delta

        # Check if value is over the end_value
        if ((self._value - self._end_value) > 0) == (self._delta > 0):
            self._value = self._end_value
        return self._value


class ConstantSchedule(Schedule):
    """Returns a constant value over the course of the schedule"""

    def __init__(self, value):
        """
        Args:
            value: the value to be returned.
        """
        self._value = value

    def get_value(self):
        return self._value

    def update(self):
        return self._value


class SwitchSchedule(Schedule):
    """Returns one value for the first part of the schedule. After the defined
    number of steps is reached, switches to returning a second value.
    """

    def __init__(self, off_value, on_value, steps):
        """
        Args:
            off_value: the value to be returned in the first part of the schedule.
            on_value: the value to be returned in the second part of the schedule.
            steps (int): the number of steps after which to switch from the off
                value to the on value.
        """

        self._steps = 0
        self._flip_step = steps
        self._off_value = off_value
        self._on_value = on_value

    def get_value(self):
        if self._steps < self._flip_step:
            return self._off_value
        else:
            return self._on_value

    def update(self):
        value = self.get_value()
        self._steps += 1
        return value


class PeriodicSchedule(Schedule):
    """Returns one value on the first step of each period of a predefined number of
        steps. Returns another value otherwise.
    """

    def __init__(self, off_value, on_value, period):
        """
        Args:
            on_value: the value to be returned on the first step of each period.
            off_value: the value to be returned for every other step in the period.
            period (int): the number of steps in the period.
        """
        self._steps = 0
        self._period = period
        self._off_value = off_value
        self._on_value = on_value

    def get_value(self):
        if (self._steps % self._period) == 0:
            return self._on_value
        else:
            return self._off_value

    def update(self):
        self._steps += 1
        return self.get_value()
