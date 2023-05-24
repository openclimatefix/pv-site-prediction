import datetime as dt
import functools

from psp.data.data_sources.nwp import NwpDataSource
from psp.data.data_sources.pv import NetcdfPvDataSource, PvDataSource
from psp.dataset import DateSplits, PvSplits, auto_date_split, split_pvs
from psp.exp_configs.base import ExpConfigBase
from psp.models.base import PvSiteModel, PvSiteModelConfig
from psp.models.recent_history import RecentHistoryModel
from psp.models.regressors.decision_trees import SklearnRegressor
from psp.typings import Horizons

PV_DATA_PATH = "data/mone/pv.nc"
PV_DATA_PATH_5MIN = "data/mone/pv_5min.nc"
NWP_DATA_PATH = "gs://solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_intermediate_version_7.zarr"


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
            nwp_tolerance=None
        )

    def make_pv_splits(self, pv_data_source: PvDataSource) -> PvSplits:
        return split_pvs(pv_data_source, pv_split=None)

    def get_date_splits(self) -> DateSplits:
        return auto_date_split(
            min_date=dt.datetime(2021, 1, 1),
            max_date=dt.datetime(2022, 12, 31),
            num_trainings=3,
            train_ratio=0.25,
        )
