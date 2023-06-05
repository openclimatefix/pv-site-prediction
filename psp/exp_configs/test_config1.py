"""Config used in tests."""

import datetime as dt

from sklearn.ensemble import HistGradientBoostingRegressor

from psp.data.data_sources.pv import NetcdfPvDataSource, PvDataSource
from psp.dataset import DateSplits, PvSplits, TestDateSplit, TrainDateSplit
from psp.exp_configs.base import ExpConfigBase
from psp.models.base import PvSiteModel, PvSiteModelConfig
from psp.models.recent_history import RecentHistoryModel
from psp.models.regressors.decision_trees import SklearnRegressor
from psp.testing import make_test_nwp_data_source
from psp.typings import Horizons

PV_DATA_PATH = "psp/tests/fixtures/pv_data.netcdf"


class ExpConfig(ExpConfigBase):
    def get_pv_data_source(self):
        return NetcdfPvDataSource(
            PV_DATA_PATH,
            id_dim_name="ss_id",
            timestamp_dim_name="timestamp",
            rename={"generation_wh": "power"},
        )

    def get_data_source_kwargs(self):
        return dict(
            pv_data_source=self.get_pv_data_source(),
            nwp_data_source=make_test_nwp_data_source(),
        )

    def get_model_config(self):
        return PvSiteModelConfig(horizons=Horizons(duration=15, num_horizons=5))

    def get_model(self, **kwargs) -> PvSiteModel:
        return RecentHistoryModel(
            config=self.get_model_config(),
            **self.get_data_source_kwargs(),
            regressor=SklearnRegressor(
                num_train_samples=20,
                sklearn_regressor=HistGradientBoostingRegressor(
                    max_iter=2,
                ),
            ),
            use_nwp=True,
        )

    def make_pv_splits(self, pv_data_source: PvDataSource) -> PvSplits:
        return PvSplits(
            train=["8215"],
            valid=["8215"],
            test=["8229"],
        )

    def get_date_splits(self) -> DateSplits:
        return DateSplits(
            train_date_splits=[TrainDateSplit(train_date=dt.datetime(2020, 1, 7), train_days=6)],
            test_date_split=TestDateSplit(
                start_date=dt.datetime(2020, 1, 8),
                end_date=dt.datetime(2020, 1, 15),
            ),
        )
