import abc
from typing import Any

from psp.data.data_sources.pv import PvDataSource
from psp.dataset import Splits
from psp.models.base import PvSiteModel


class ExpConfigBase(abc.ABC):
    """Defines the interface of an experiment config."""

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
    def get_model(self) -> PvSiteModel:
        """Get the model"""
        pass

    @abc.abstractmethod
    def make_dataset_splits(self, pv_data_source: PvDataSource) -> Splits:
        """Make the dataset splits from the pv data source."""
        pass
