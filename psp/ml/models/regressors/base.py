"""Define the interface of a Regressor, which is used in models."""

import abc
from typing import Iterator

import numpy as np

from psp.ml.typings import Batch, BatchedFeatures


class Regressor(abc.ABC):
    @abc.abstractmethod
    def train(
        self, train_iter: Iterator[Batch], valid_iter: Iterator[Batch], batch_size: int
    ):
        pass

    @abc.abstractmethod
    def predict(self, features: BatchedFeatures) -> np.ndarray:
        pass
