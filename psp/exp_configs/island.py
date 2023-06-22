"""Main config for the "island" use-case."""

import datetime as dt

import xarray as xr
from psp.data_sources.nwp import NwpDataSource
from psp.data_sources.pv import NetcdfPvDataSource, PvDataSource
from psp.dataset import PvSplits, auto_date_split, split_pvs
from psp.exp_configs.base import ExpConfigBase
from psp.models.base import PvSiteModel, PvSiteModelConfig
from psp.models.recent_history import RecentHistoryModel
from psp.models.regressors.decision_trees import SklearnRegressor
from psp.typings import Horizons
from sklearn.ensemble import HistGradientBoostingRegressor

PV_TARGET_DATA_PATH = (
    "./normal.nc"
    # "./tweaked.nc"
    # "/mnt/storage_b/data/ocf/solar_pv_nowcasting/clients/island/pv_hourly_v8_hacky.nc"
    # "/mnt/storage_b/data/ocf/solar_pv_nowcasting/clients/island/data_hourly_MW_dstfix_clean_v2.nc"
    # "/mnt/storage_b/data/ocf/solar_pv_nowcasting/clients/island/enemalta_pv_hourly_v5.nc"
)
NWP_PATH = "/mnt/storage_b/data/ocf/solar_pv_nowcasting/clients/island/nwp_v8.zarr"


def _get_capacity(data: xr.Dataset) -> float:
    # Use the "capacity" data variable.
    # There is one capacity per timestamp, let's take the first one arbitrarily; hopefully there
    # is not a big variation of capacity in our history of a few days!
    return float(data["capacity"].values[0])


class ExpConfig(ExpConfigBase):
    def get_pv_data_source(self):
        return NetcdfPvDataSource(
            PV_TARGET_DATA_PATH,
            # id_dim_name="id",
            # timestamp_dim_name='datetimeUTC',
            timestamp_dim_name="timestamp",
            # rename={
            # "Hourly PV Generated Units (MW)": "power",
            # "Total Max Capacity (MW)": "capacity",
            # },
        )

    def get_data_source_kwargs(self):
        return dict(
            pv_data_source=NetcdfPvDataSource(
                PV_TARGET_DATA_PATH,
                # id_dim_name="id",
                # timestamp_dim_name='datetimeUTC',
                timestamp_dim_name="timestamp",
                # rename={
                # "Hourly PV Generated Units (MW)": "power",
                # "Total Max Capacity (MW)": "capacity",
                # },
                lag_minutes=5 * 24 * 60,
            ),
            nwp_data_source=NwpDataSource(
                NWP_PATH,
                coord_system=4326,
                x_dim_name="latitude",
                y_dim_name="longitude",
                x_is_ascending=False,
            ),
        )

    def get_model_config(self):
        return PvSiteModelConfig(
            horizons=Horizons(
                duration=60,
                num_horizons=48,
            )
        )

    def get_model(self, **kwargs) -> PvSiteModel:
        return RecentHistoryModel(
            self.get_model_config(),
            **self.get_data_source_kwargs(),
            regressor=SklearnRegressor(
                num_train_samples=4096,
                normalize_targets=True,
                sklearn_regressor=HistGradientBoostingRegressor(
                    loss="absolute_error",
                    random_state=1234,
                    max_iter=200,
                ),
            ),
            use_nwp=True,
            normalize_features=True,
            capacity_getter=_get_capacity,
            use_capacity_as_feature=False,
            num_days_history=12,
        )

    def make_pv_splits(self, pv_data_source: PvDataSource) -> PvSplits:
        return split_pvs(
            pv_data_source,
            pv_split=None,
        )

    def get_date_splits(self):
        return auto_date_split(
            test_start_date=dt.datetime(2020, 10, 14),
            test_end_date=dt.datetime(2022, 10, 14),
            num_trainings=8,
            train_days=365 * 2,
            min_train_date=dt.datetime(2018, 11, 2),
        )
