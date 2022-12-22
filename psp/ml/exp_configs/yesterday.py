import functools

from psp.data.data_sources.pv import NetcdfPvDataSource
from psp.ml.models.base import PvSiteModel, PvSiteModelConfig
from psp.ml.models.yesterday import SetupConfig, YesterdayPvSiteModel

PV_DATA_PATH = "data/5min_2.netcdf"

# TODO define
ExpConfigBase = object


class ExpConfig(ExpConfigBase):
    @functools.cache
    def get_pv_data_source(self):
        return NetcdfPvDataSource(PV_DATA_PATH)

    @functools.cache
    def get_model_setup_config(self):
        return SetupConfig(data_source=self.get_pv_data_source())

    @functools.cache
    def _get_model_config(self):
        interval_size = 15
        interval_starts = [0.0, 30, 120, 24 * 60, 48 * 60]
        future_intervals = [(s, s + interval_size) for s in interval_starts]
        return PvSiteModelConfig(future_intervals=future_intervals, blackout=0)

    @functools.cache
    def get_model(self) -> PvSiteModel:
        return YesterdayPvSiteModel(
            self._get_model_config(), self.get_model_setup_config()
        )
