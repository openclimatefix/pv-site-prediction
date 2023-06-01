import datetime as dt
import functools

from psp.data.data_sources.nwp import NwpDataSource
from psp.data.data_sources.pv import NetcdfPvDataSource, PvDataSource
from psp.dataset import PvSplits, auto_date_split, split_pvs
from psp.exp_configs.base import ExpConfigBase
from psp.models.base import PvSiteModel, PvSiteModelConfig
from psp.models.recent_history import RecentHistoryModel
from psp.models.regressors.decision_trees import SklearnRegressor
from psp.typings import Horizons

_PREFIX = "/mnt/storage_b/data/ocf/solar_pv_nowcasting/clients/mone"
PV_DATA_PATH = _PREFIX + "/pv_v3.nc"
PV_DATA_PATH_5MIN = _PREFIX + "/pv_v3_5min.nc"
NWP_DATA_PATH = (
    "/mnt/storage_ssd_8tb/data/ocf/solar_pv_nowcasting"
    "/nowcasting_dataset_pipeline/NWP/UK_Met_Office/UKV/zarr"
    "/UKV_intermediate_version_7.zarr"
)


class ExpConfig(ExpConfigBase):
    @functools.cache
    def get_pv_data_source(self):
        return NetcdfPvDataSource(
            PV_DATA_PATH,
        )

    @functools.cache
    def get_data_source_kwargs(self):
        return dict(
            pv_data_source=NetcdfPvDataSource(PV_DATA_PATH_5MIN),
            nwp_data_source=NwpDataSource(
                NWP_DATA_PATH,
                coord_system=27700,
                time_dim_name="init_time",
                value_name="UKV",
                y_is_ascending=False,
            ),
        )

    def get_model_config(self) -> PvSiteModelConfig:
        return PvSiteModelConfig(horizons=Horizons(duration=30, num_horizons=48))

    def get_model(self) -> PvSiteModel:
        return RecentHistoryModel(
            config=self.get_model_config(),
            **self.get_data_source_kwargs(),
            regressor=SklearnRegressor(
                num_train_samples=1024,
                normalize_targets=True,
            ),
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
            normalize_features=True,
            use_inferred_meta=False,
            nwp_tolerance="168h",
        )

    def make_pv_splits(self, pv_data_source: PvDataSource) -> PvSplits:
        return split_pvs(pv_data_source, pv_split=None)

    def get_date_splits(self):
        return auto_date_split(
            test_start_date=dt.datetime(2021, 7, 1),
            test_end_date=dt.datetime(2022, 12, 31),
            # Using 3 trainings because the NWP data situation changes over time. When we have NWP
            # data across the board, 1 training will probably be enough.
            num_trainings=3,
            train_days=30 * 6,
        )
