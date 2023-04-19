"""Config used in tests."""

import functools

from psp.data.data_sources.pv import NetcdfPvDataSource
from psp.models.base import PvSiteModel, PvSiteModelConfig
from psp.models.recent_history import RecentHistoryModel, SetupConfig
from psp.models.regressors.decision_trees import SklearnRegressor
from psp.typings import Horizons

PV_DATA_PATH = "psp/tests/fixtures/pv_data.netcdf"


class ExpConfig:
    @functools.cache
    def get_pv_data_source(self):
        return NetcdfPvDataSource(
            PV_DATA_PATH,
            id_dim_name="ss_id",
            timestamp_dim_name="timestamp",
            rename={"generation_wh": "power"},
        )

    @functools.cache
    def get_model_setup_config(self):
        return SetupConfig(pv_data_source=self.get_pv_data_source(), nwp_data_source=None)

    @functools.cache
    def _get_model_config(self):
        return PvSiteModelConfig(horizons=Horizons(duration=15, num_horizons=5), blackout=0)

    @functools.cache
    def get_model(self) -> PvSiteModel:
        return RecentHistoryModel(
            self._get_model_config(),
            self.get_model_setup_config(),
            regressor=SklearnRegressor(
                num_train_samples=10,
            ),
            use_nwp=False,
        )
