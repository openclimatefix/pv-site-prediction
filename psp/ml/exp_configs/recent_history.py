import functools

from psp.data.data_sources.nwp import NwpDataSource
from psp.data.data_sources.pv import NetcdfPvDataSource
from psp.ml.models.base import PvSiteModel, PvSiteModelConfig
from psp.ml.models.recent_history import RecentHistoryModel, SetupConfig
from psp.ml.models.regressors.decision_trees import ForestRegressor

PV_DATA_PATH = "data/5min.netcdf"
NWP_DATA_PATH = (
    "gs://solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_intermediate_version_3.zarr"
)

# TODO define
ExpConfigBase = object


class ExpConfig(ExpConfigBase):
    @functools.cache
    def get_pv_data_source(self):
        return NetcdfPvDataSource(PV_DATA_PATH)

    @functools.cache
    def get_model_setup_config(self):
        return SetupConfig(
            pv_data_source=self.get_pv_data_source(),
            nwp_data_source=NwpDataSource(NWP_DATA_PATH),
        )

    @functools.cache
    def _get_model_config(self):
        interval_size = 15
        # Start of the inverval in hours.
        interval_starts = [0.0, 0.5, 2, 4, 6, 8, 12, 18, 24, 30, 36, 42, 48]
        future_intervals = [(s * 60, s * 60 + interval_size) for s in interval_starts]
        return PvSiteModelConfig(future_intervals=future_intervals, blackout=0)

    @functools.cache
    def get_model(self) -> PvSiteModel:
        return RecentHistoryModel(
            self._get_model_config(),
            self.get_model_setup_config(),
            regressor=ForestRegressor(num_train_samples=4096),
            use_nwp=True,
        )
