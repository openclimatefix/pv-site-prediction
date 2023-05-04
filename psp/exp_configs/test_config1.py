"""Config used in tests."""

import functools

from sklearn.ensemble import HistGradientBoostingRegressor

from psp.data.data_sources.pv import NetcdfPvDataSource, PvDataSource
from psp.dataset import Splits, split_train_test
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
    def _get_model_config(self):
        return PvSiteModelConfig(horizons=Horizons(duration=15, num_horizons=5))

    @functools.cache
    def get_model(self) -> PvSiteModel:
        return RecentHistoryModel(
            config=self._get_model_config(),
            **self.get_data_source_kwargs(),
            regressor=SklearnRegressor(
                num_train_samples=10,
                sklearn_regressor=HistGradientBoostingRegressor(
                    max_iter=2,
                ),
            ),
            use_nwp=False,
        )

    def make_dataset_splits(self, pv_data_source: PvDataSource) -> Splits:
        return split_train_test(pv_data_source)
