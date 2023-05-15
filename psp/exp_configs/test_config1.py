"""Config used in tests."""

import datetime as dt
import functools

from sklearn.ensemble import HistGradientBoostingRegressor

from psp.data.data_sources.pv import NetcdfPvDataSource, PvDataSource
from psp.dataset import DateSplits, PvSplits
from psp.exp_configs.base import ExpConfigBase
from psp.models.base import PvSiteModel, PvSiteModelConfig
from psp.models.recent_history import RecentHistoryModel
from psp.models.regressors.decision_trees import SklearnRegressor
from psp.typings import Horizons

PV_DATA_PATH = "psp/tests/fixtures/pv_data.netcdf"


class ExpConfig(ExpConfigBase):
    @functools.cache
    def get_pv_data_source(self):
        return NetcdfPvDataSource(
            PV_DATA_PATH,
            id_dim_name="ss_id",
            timestamp_dim_name="timestamp",
            rename={"generation_wh": "power"},
        )

    @functools.cache
    def get_data_source_kwargs(self):
        return dict(pv_data_source=self.get_pv_data_source(), nwp_data_source=None)

    @functools.cache
    def get_model_config(self):
        return PvSiteModelConfig(horizons=Horizons(duration=15, num_horizons=5))

    @functools.cache
    def get_model(self) -> PvSiteModel:
        return RecentHistoryModel(
            config=self.get_model_config(),
            **self.get_data_source_kwargs(),
            regressor=SklearnRegressor(
                num_train_samples=10,
                sklearn_regressor=HistGradientBoostingRegressor(
                    max_iter=2,
                ),
            ),
            use_nwp=False,
        )

    def make_pv_splits(self, pv_data_source: PvDataSource) -> PvSplits:
        return PvSplits(
            train=["8215"],
            valid=["8215"],
            test=["8229"],
        )

    def get_date_splits(self) -> DateSplits:
        return DateSplits(
            train_dates=[dt.datetime(2020, 1, 7)],
            num_train_days=6,
            num_test_days=7,
        )
