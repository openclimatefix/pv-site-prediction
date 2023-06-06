import abc
from typing import Any

import numpy as np

from psp.data_sources.pv import PvDataSource
from psp.dataset import DateSplits
from psp.models.base import PvSiteModel, PvSiteModelConfig


class ExpConfigBase(abc.ABC):
    """Defines the interface of an experiment config."""

    @abc.abstractmethod
    def get_model_config(self) -> PvSiteModelConfig:
        pass

    @abc.abstractmethod
    def get_pv_data_source(self) -> PvDataSource:
        """Get the PV data source used for the targets."""
        pass

    @abc.abstractmethod
    def get_data_source_kwargs(self) -> dict[str, Any]:
        """
        Get the keyword arguments that we pass to the `set_data_sources` method of the
        `PvSiteModel`.
        """
        pass

    @abc.abstractmethod
    def get_model(self, *, random_state: np.random.RandomState | None = None) -> PvSiteModel:
        """Get the model"""
        pass

    @abc.abstractmethod
    def make_pv_splits(self, pv_data_source: PvDataSource):
        """Make the dataset splits from the pv data source."""
        pass

    @abc.abstractmethod
    def get_date_splits(self) -> DateSplits:
        pass
