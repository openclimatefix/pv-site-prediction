"""Config used to train a model based on the `uk_pv` dataset."""

import datetime as dt

import numpy as np

from psp.data_sources.nwp import NwpDataSource
from psp.data_sources.pv import NetcdfPvDataSource, PvDataSource
from psp.data_sources.satellite import SatelliteDataSource
from psp.dataset import PvSplits, auto_date_split, split_pvs
from psp.exp_configs.base import ExpConfigBase
from psp.models.base import PvSiteModel, PvSiteModelConfig
from psp.models.recent_history import RecentHistoryModel
from psp.models.regressors.decision_trees import SklearnRegressor
from psp.typings import Horizons

PV_DATA_PATH = "/mnt/storage_b/data/ocf/solar_pv_nowcasting/clients/dual_ax/PV/dual_ax_tracker_15min.nc"

NWP_DATA_PATHS = [
    (
        "/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP"
        f"/UK_Met_Office/UKV/zarr/UKV_{year}_NWP.zarr"
    )
    for year in range(2018, 2023)
]

EXC_PATH = [
    (
        "/mnt/storage_b/data/ocf/solar_pv_nowcasting/experimental/Excarta/"
        f"merged_zarrs/test_3_temp/excarta_{year}.zarr"
    )
    for year in [2019, 2020, 2021, 2022]
]

ECMWF_PATH = [
    (
        "/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/"
        f"NWP/ECMWF/uk/year_merged/{year}.zarr"
    )
    for year in [2020, 2021, 2022]
]

SATELLITE_DATA_PATHS = [
    (
        f"gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v4/{year}_nonhrv.zarr"
    )
    for year in range(2018, 2022)
]

def _get_capacity(d):
    # Use 0.99 quantile over the history window, fallback on the capacity as defined
    # in the metadata.
    value = float(d["power"].quantile(0.99))
    if not np.isfinite(value):
        value = float(d.coords["capacity"].values)
    return value


def _get_tilt(d):
    tilt_values = d["tilt"].values
    return tilt_values


def _get_orientation(d):
    orientation_values = d["orientation"].values
    return orientation_values


class ExpConfig(ExpConfigBase):
    def get_pv_data_source(self):
        return NetcdfPvDataSource(
            PV_DATA_PATH,
            id_dim_name="pv_id",
            timestamp_dim_name="ts",
            lag_minutes = 5,
        )

    def get_data_source_kwargs(self):
        return dict(
            pv_data_source=self.get_pv_data_source(),
            nwp_data_sources={
                "UKV": NwpDataSource(
                    NWP_DATA_PATHS,
                    coord_system=27700,
                    time_dim_name="init_time",
                    value_name="UKV",
                    y_is_ascending=False,
                    # cache_dir=".nwp_cache",
                    # Those are the variables available in our prod environment.
                    variables=[
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
                ),
                # "EXC": NwpDataSource(
                #     EXC_PATH,
                #     coord_system=4326,
                #     x_dim_name="latitude",
                #     y_dim_name="longitude",
                #     time_dim_name="ts",
                #     x_is_ascending=True,
                #     y_is_ascending=True,
                #     lag_minutes=7 * 60,
                #     tolerance="168h",
                # ),
                "ECMWF": NwpDataSource(
                    ECMWF_PATH,
                    coord_system=4326,
                    x_dim_name="latitude",
                    y_dim_name="longitude",
                    time_dim_name="init_time",
                    value_name="UKV",
                    x_is_ascending=True,
                    y_is_ascending=False,
                    lag_minutes=6 * 60,
                    tolerance="168h",
                ),
            },
            # satellite_data_sources={
            #     "EUMETSAT": SatelliteDataSource(
            #         SATELLITE_DATA_PATHS,
            #         x_is_ascending=False,
            #     ),
            # },
        )

    def _get_model_config(self) -> PvSiteModelConfig:
        return PvSiteModelConfig(horizons=Horizons(duration=15, num_horizons= 4))

    def get_model(self, *, random_state: np.random.RandomState | None = None) -> PvSiteModel:
        kwargs = self.get_data_source_kwargs()
        return RecentHistoryModel(
            config=self._get_model_config(),
            **kwargs,
            regressor=SklearnRegressor(
                num_train_samples=4096,
                normalize_targets=True,
            ),
            random_state=random_state,
            normalize_features=True,
            capacity_getter=_get_capacity,
            pv_dropout=0.1,
            nwp_dropout=0.1,
        )

    def make_pv_splits(self, pv_data_source: PvDataSource) -> PvSplits:
        return split_pvs(pv_data_source)

    def get_date_splits(self):
        return auto_date_split(
            test_start_date=dt.datetime(2023, 1, 1),
            test_end_date=dt.datetime(2023, 11, 19),
            train_days=365 * 2,
            min_train_date=dt.datetime(2021, 9, 18),
        )
