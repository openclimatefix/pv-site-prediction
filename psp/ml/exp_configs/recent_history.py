import functools

from psp.data.data_sources.nwp import NwpDataSource
from psp.data.data_sources.pv import NetcdfPvDataSource
from psp.ml.models.base import PvSiteModel, PvSiteModelConfig
from psp.ml.models.recent_history import RecentHistoryModel, SetupConfig
from psp.ml.models.regressors.decision_trees import ForestRegressor

PV_DATA_PATH = "data/5min.netcdf"
NWP_DATA_PATH = (
    "gs://solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_intermediate_version_7.zarr"
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
        delta = 15.0
        horizons = [(i * delta, (i + 1) * delta) for i in range(48 * 4)]
        return PvSiteModelConfig(horizons=horizons, blackout=0)

    @functools.cache
    def get_model(self) -> PvSiteModel:
        return RecentHistoryModel(
            self._get_model_config(),
            self.get_model_setup_config(),
            regressor=ForestRegressor(num_train_samples=4096),
            use_nwp=True,
            # Those are the variables available in our prod environment.
            nwp_variables=[
                "si10",
                "vis",
                # "r2",
                "t",
                "prate",
                # "sd",
                "dlwrf",
                "dswrf",
                "hcc",
                "mcc",
                "lcc",
            ],
        )
