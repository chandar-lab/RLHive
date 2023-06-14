import abc
from typing import Generic, TypeVar, Union

from hive.utils.registry import registry
from hive.utils.utils import Counter

T = TypeVar("T")
Numeric = TypeVar("Numeric", int, float)
Int = Union[int, Counter]


class Schedule(abc.ABC, Generic[T]):
    @abc.abstractmethod
    def __call__(self, step: Int) -> T:
        """Queries the value of the schedule at a given step."""
        pass

    @classmethod
    def type_name(cls):
        return "schedule"


class LinearSchedule(Schedule[Numeric]):
    """Defines a linear schedule between two values over some number of steps.

    When queried with a step value greater than `steps`, the schedule stays at the
    end value.
    """

    def __init__(self, init_value: Numeric, end_value: Numeric, steps: int):
        """
        Args:
            init_value (int | float): Starting value for schedule.
            end_value (int | float): End value for schedule.
            steps (int): Number of steps for schedule. Should be positive.
        """
        steps = max(int(steps), 1)
        self._init_value = init_value
        self._steps = steps
        self._end_value = end_value
        self._delta = (end_value - init_value) / steps

    def __call__(self, step):
        if self._end_value > self._init_value:
            return min(self._init_value + self._delta * step, self._end_value)
        else:
            return max(self._init_value + self._delta * step, self._end_value)

    def __repr__(self):
        return (
            f"<class {type(self).__name__}"
            f" init_value={self._init_value}"
            f" steps={self._steps}"
            f" end_value={self._end_value}>"
        )


class ConstantSchedule(Schedule[T]):
    """Returns a constant value over the course of the schedule"""

    def __init__(self, value: T):
        """
        Args:
            value: The value to be returned.
        """
        self._value = value

    def __call__(self, step):
        return self._value

    def __repr__(self):
        return f"<class {type(self).__name__} value={self._value}>"


class SwitchSchedule(Schedule[T]):
    """Returns one value for the first part of the schedule. After the defined
    number of steps is reached, switches to returning a second value.
    """

    def __init__(self, off_value: T, on_value: T, steps: int):
        """
        Args:
            off_value: The value to be returned in the first part of the schedule.
            on_value: The value to be returned in the second part of the schedule.
            steps (int): The number of steps after which to switch from the off
                value to the on value.
        """

        self._flip_step = steps
        self._off_value = off_value
        self._on_value = on_value

    def __call__(self, step):
        if step < self._flip_step:
            return self._off_value
        else:
            return self._on_value

    def __repr__(self):
        return (
            f"<class {type(self).__name__}"
            f" off_value={self._off_value}"
            f" on_value={self._on_value}"
            f" flip_step={self._flip_step}>"
        )


class DoublePeriodicSchedule(Schedule[T]):
    """Returns off value for off period, then switches to returning on value for on
    period. Alternates between the two.
    """

    def __init__(self, off_value: T, on_value: T, off_period: int, on_period: int):
        """
        Args:
            on_value: The value to be returned for the on period.
            off_value: The value to be returned for the off period.
            on_period (int): the number of steps in the on period.
            off_period (int): the number of steps in the off period.
        """
        self._off_period = off_period
        self._total_period = self._off_period + on_period
        self._off_value = off_value
        self._on_value = on_value

    def __call__(self, step):
        if (step % self._total_period) == 0:
            return self._on_value
        else:
            return self._off_value

    def __repr__(self):
        return (
            f"<class {type(self).__name__}"
            f" off_value={self._off_value}"
            f" on_value={self._on_value}"
            f" off_period={self._off_period}"
            f" on_period={self._total_period - self._off_period}>"
        )


class PeriodicSchedule(DoublePeriodicSchedule[T]):
    """Returns one value on the first step of each period of a predefined number of
    steps. Returns another value otherwise.
    """

    def __init__(self, off_value: T, on_value: T, period: int):
        """
        Args:
            on_value: The value to be returned on the first step of each period.
            off_value: The value to be returned for every other step in the period.
            period (int): The number of steps in the period.
        """
        super().__init__(off_value, on_value, period - 1, 1)

    def __repr__(self):
        return (
            f"<class {type(self).__name__}"
            f" off_value={self._off_value}"
            f" on_value={self._on_value}"
            f" period={self._off_period + 1}>"
        )


registry.register_classes(
    {
        "LinearSchedule": LinearSchedule,
        "ConstantSchedule": ConstantSchedule,
        "SwitchSchedule": SwitchSchedule,
        "PeriodicSchedule": PeriodicSchedule,
        "DoublePeriodicSchedule": DoublePeriodicSchedule,
    },
)
