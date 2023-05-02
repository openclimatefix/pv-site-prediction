"""Main config for the "island" use-case."""

import datetime as dt
import functools

from psp.data.data_sources.nwp import NwpDataSource
from psp.data.data_sources.pv import NetcdfPvDataSource, PvDataSource
from psp.dataset import Splits, split_train_test
from psp.exp_configs.base import ExpConfigBase
from psp.models.base import PvSiteModel, PvSiteModelConfig
from psp.models.recent_history import RecentHistoryModel
from psp.models.regressors.decision_trees import SklearnRegressor
from psp.typings import Horizons

PV_DATA_PATH = "data/island/data_15min_FIN.nc"
NWP_PATH = "data/island/nwp_simon_v5.nwp"


class ExpConfig(ExpConfigBase):
    @functools.cache
    def get_pv_data_source(self):
        return NetcdfPvDataSource(
            PV_DATA_PATH,
            id_dim_name="id",
            timestamp_dim_name="datetimeUTC",
            rename={
                "Power at point in time MW": "power",
                "Total Installed Capacity MWp": "capacity",
            },
        )

    @functools.cache
    def get_data_source_kwargs(self):
        return dict(
            pv_data_source=self.get_pv_data_source(),
            nwp_data_source=NwpDataSource(
                NWP_PATH,
                coord_system=4326,
                x_dim_name="latitude",
                y_dim_name="longitude",
            ),
        )

    @functools.cache
    def _get_model_config(self):
        return PvSiteModelConfig(horizons=Horizons(duration=15, num_horizons=48 * 4), blackout=0)

    @functools.cache
    def get_model(self) -> PvSiteModel:
        return RecentHistoryModel(
            self._get_model_config(),
            **self.get_data_source_kwargs(),
            regressor=SklearnRegressor(
                num_train_samples=2048,
                normalize_targets=True,
            ),
            use_nwp=True,
            normalize_features=True,
            use_inferred_meta=False,
            use_data_capacity=True,
        )

    def make_dataset_splits(self, pv_data_source: PvDataSource) -> Splits:
        return split_train_test(
            pv_data_source,
            pv_split=None,
            # train_start = datetime(2018, 1, 1)
            train_start=dt.datetime(2021, 4, 12),
            # Leaving a couple of days at the end to be safe.
            train_end=dt.datetime(2022, 4, 12),
            test_start=dt.datetime(2022, 4, 13),
            test_end=dt.datetime(2023, 1, 1),
        )
