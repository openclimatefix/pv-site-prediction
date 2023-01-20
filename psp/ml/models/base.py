"""Base classes for the PV site ml."""

import abc
import dataclasses
from typing import Any, Iterator

from psp.ml.typings import Batch, Features, FutureIntervals, X, Y


@dataclasses.dataclass
class PvSiteModelConfig:
    """Model meta data that all models must define."""

    future_intervals: FutureIntervals
    # Blackout in minutes.
    # This is a window of time before the timestamp at which we don't have access to data.
    # This is to simulate a delay in the availability of the data, which can happen in production.
    blackout: int


class PvSiteModel(abc.ABC):
    """Abstract interface for our models."""

    def __init__(self, config: PvSiteModelConfig, setup_config: Any):
        self._config = config

    @abc.abstractmethod
    def predict_from_features(self, features: Features) -> Y:
        """Predict the output from the features.

        Useful if the features were already computed, or to leverage
        computing features in parallel separately.
        """
        pass

    def predict(self, x: X) -> Y:
        """Predict the output from the input.

        This is what should be called in production.
        """
        features = self.get_features(x)
        return self.predict_from_features(features)

    @abc.abstractmethod
    def get_features(self, x: X) -> Features:
        """Compute features for the model.

        This step will be run in parallel by our data pipelines.
        """
        pass

    @property
    def config(self):
        return self._config

    def train(
        self, train_iter: Iterator[Batch], valid_iter: Iterator[Batch], batch_size: int
    ) -> None:
        """Train the model."""
        pass

    def setup(self, setup_config: Any):
        """Set up the model after initialization or deserialization.

        For instance defining data sources.
        """
        pass

    def get_state(self):
        """Return the necessary fields of the class for serialization.

        This is used by `psp.ml.serialization` to load and save the model.

        We need a different hook than `__getstate__` because sometimes we want to customize the
        model serialization and the default pickling in different ways. An example of this is that
        we want pytorch's `DataLoader` to pickle our model alongside its data sources, but when we
        serialize a model, we don't want them.

        This is meant to be overridden in children classes if a custom behaviour is needed.
        """
        return self.__dict__.copy()
