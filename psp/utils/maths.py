from typing import TypeVar, overload

import numpy as np
import xarray as xr

T = TypeVar("T", bound=np.ndarray | xr.DataArray)


@overload
def safe_div(num: float, den: float) -> float:
    ...


@overload
def safe_div(num: T, den: T | float) -> T:
    ...


@overload
def safe_div(num: T | float, den: T) -> T:
    ...


def safe_div(num, den):
    return np.divide(num, den, out=np.zeros_like(num), where=den != 0)


class MeanAggregator:
    """Utility class to track the mean of a value.

    Useful to track losses while training.
    """

    def __init__(self):
        self.reset()

    def add(self, value: float, n: int = 1):
        self._total += value
        self._n += n

    def mean(self):
        if self._n == 0:
            return 0.0
        return self._total / self._n

    def reset(self):
        self._total = 0.0
        self._n = 0
