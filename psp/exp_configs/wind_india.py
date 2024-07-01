"""Main config for the "island" use-case."""

import datetime as dt

import numpy as np
import xarray as xr
import glob
import os

from psp.data_sources.nwp import NwpDataSource
from psp.data_sources.pv import NetcdfPvDataSource, PvDataSource
from psp.dataset import PvSplits, auto_date_split, split_pvs
from psp.exp_configs.base import ExpConfigBase
from psp.models.base import PvSiteModel, PvSiteModelConfig
from psp.models.recent_history import RecentHistoryModel
from psp.models.regressors.decision_trees import SklearnRegressor
from psp.typings import Horizons

PV_TARGET_DATA_PATH = "/mnt/storage_ssd_4tb/india_wind_data_format_for_site_model_v5.nc"
# GFS_NWP_PATH = "/mnt/storage_b/data/ocf/solar_pv_nowcasting/clients/island/5y_nwp_3_to_48.zarr"
# EXC_PATH = [
#     (
#         "/mnt/storage_b/data/ocf/solar_pv_nowcasting/experimental/Excarta/"
#         f"merged_zarrs/test_3_temp/excarta_{year}.zarr"
#     )
#     for year in [2019, 2020, 2021, 2022]  # , 2019, 2020, 2021, 2022]
# ]

base_path_ecmwf = "/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/ECMWF/nw-india/zarr/"
zarr_files_ecmwf = glob.glob(os.path.join(base_path_ecmwf, "*.zarr.zip"))

ECMWF_PATH = zarr_files_ecmwf


def _get_capacity(data: xr.Dataset) -> float:
    # Use the "capacity" data variable.
    # There is one capacity per timestamp, let's take the first one arbitrarily; hopefully there
    # is not a big variation of capacity in our history of a few days!
    return float(data.coords["capacity"].values)


class ExpConfig(ExpConfigBase):
    def get_pv_data_source(self):
        return NetcdfPvDataSource(
            PV_TARGET_DATA_PATH,
            id_dim_name="pv_id",
            timestamp_dim_name="ts",
            # rename={"capacity_mw": "capacity"},
        )

    def get_data_source_kwargs(self):
        return dict(
            pv_data_source=NetcdfPvDataSource(
                PV_TARGET_DATA_PATH,
                lag_minutes=5,
            ),
            nwp_data_sources={
                "ECMWF": NwpDataSource(
                    ECMWF_PATH,
                    coord_system=4326,
                    x_dim_name="latitude",
                    y_dim_name="longitude",
                    time_dim_name="init_time",
                    value_name="ECMWF_NW-INDIA",
                    x_is_ascending=True,
                    y_is_ascending=False,
                    lag_minutes=6 * 60,
                    tolerance="168h",
                ),
            },
        )

    def get_model_config(self):
        return PvSiteModelConfig(
            horizons=Horizons(
                duration=15,
                num_horizons=48 * 4,
            )
        )

    def get_model(self, random_state: np.random.RandomState | None = None) -> PvSiteModel:
        return RecentHistoryModel(
            self.get_model_config(),
            **self.get_data_source_kwargs(),
            regressor=SklearnRegressor(
                num_train_samples=4096,
                normalize_targets=False,
            ),
            random_state=random_state,
            normalize_features=False,
            capacity_getter=_get_capacity,
            use_capacity_as_feature=False,
            num_days_history=12,
            nwp_dropout=0.1,
        )

    def make_pv_splits(self, pv_data_source: PvDataSource) -> PvSplits:
        return split_pvs(
            pv_data_source,
            pv_split=None,
        )

    def get_date_splits(self):
        return auto_date_split(
            test_start_date=dt.datetime(2022, 12, 1),
            test_end_date=dt.datetime(2023, 12, 31),
            num_trainings=1,
            train_days=365 * 3,
            # Min date because of NWP not available at the beginning of the PV data.
            min_train_date=dt.datetime(2019, 1, 1),
            step_minutes=60,
        )
