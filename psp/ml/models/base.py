"""Base classes for the PV site ml."""

import abc
from typing import Any, Generic, Mapping, TypeVar

from psp.ml.typings import FutureIntervals, X, Y

FeaturesType = Mapping[str, Any]

F = TypeVar("F", bound=FeaturesType)


class PvSiteModel(abc.ABC, Generic[F]):
    def __init__(self, future_intervals: FutureIntervals):
        self._future_intervals = future_intervals

    @abc.abstractmethod
    def _predict_from_features(self, x: X, features: F) -> Y:
        pass

    def predict(self, x: X, features: F | None) -> Y:
        """Predict the input, given an input.

        Arguments
            x: The input.
            features: Optional. If features are provided, we will simply skip the step where we
                compute them. This is useful if we want to compute features in parallel during
                training and evaluation.
        """
        if features is None:
            features = self.get_features(x)
        return self._predict_from_features(x, features)

    @abc.abstractmethod
    def get_features(self, x: X) -> F:
        pass

    @property
    def future_intervals(self):
        return self._future_intervals
