from typing import overload

import numpy as np


@overload
def safe_div(num: float, den: float) -> float:
    ...


@overload
def safe_div(num: np.ndarray, den: np.ndarray | float) -> np.ndarray:
    ...


@overload
def safe_div(num: np.ndarray | float, den: np.ndarray) -> np.ndarray:
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
