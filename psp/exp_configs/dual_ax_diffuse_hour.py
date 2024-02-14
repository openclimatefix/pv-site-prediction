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

# import multiprocessing
# import xgboost as xgb

PV_DATA_PATH = "/mnt/leonardo/storage_b/data/ocf/solar_pv_nowcasting/clients/dual_ax/PV/dual_ax_tracker_in_diffuse_15min_mid_kw.nc"
# NWP_DATA_PATH = "gs://solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_intermediate_version_7.zarr"
NWP_DATA_PATHS = [
    (
        "/mnt/leonardo/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP"
        f"/UK_Met_Office/UKV/zarr/UKV_{year}_NWP.zarr"
    )
    for year in range(2018, 2022)
]

ECMWF_PATH = [
    (
        "/mnt/leonardo/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/"
        f"NWP/ECMWF/uk/year_merged/{year}.zarr"
    )
    for year in [2020, 2021, 2022, 2023]
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
        )

    def get_data_source_kwargs(self):
        return dict(
            pv_data_source=self.get_pv_data_source(),
            nwp_data_sources={
                "ECMWF": NwpDataSource(
                    ECMWF_PATH,
                    coord_system=4326,
                    x_dim_name="latitude",
                    y_dim_name="longitude",
                    time_dim_name="init_time",
                    value_name="ECMWF_UK",
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
        return PvSiteModelConfig(horizons=Horizons(duration=15, num_horizons=4))

    def get_model(self, *, random_state: np.random.RandomState | None = None) -> PvSiteModel:
        kwargs = self.get_data_source_kwargs()
        return RecentHistoryModel(
            config=self._get_model_config(),
            **kwargs,
            regressor=SklearnRegressor(
                num_train_samples=2000,
                normalize_targets=True,
                #
                # We have done some tests with xgboost and keep this as an example but note that we
                # haven't added xgboost to our list of dependencies.
                #
                # sklearn_regressor=xgb.XGBRegressor(
                #     objective='reg:pseudohubererror',
                #     eta=0.1,
                #     n_estimators=200,
                #     max_depth=5,
                #     min_child_weight=20,
                #     tree_method='hist',
                #     n_jobs=multiprocessing.cpu_count() // 2,
                # ),
            ),
            random_state=random_state,
            normalize_features=True,
            capacity_getter=_get_capacity,
            tilt_getter=_get_tilt,
            orientation_getter=_get_orientation,
            pv_dropout=0.1,
            nwp_dropout=0.1,
            recent_power_minutes=30,
        )

    def make_pv_splits(self, pv_data_source: PvDataSource) -> PvSplits:
        return split_pvs(pv_data_source, pv_split=None)

    def get_date_splits(self):
        return auto_date_split(
            test_start_date=dt.datetime(2023, 11, 18),
            test_end_date=dt.datetime(2023, 12, 20),
            num_trainings=1,
            train_days=365 * 2,
            min_train_date=dt.datetime(2021, 1, 18),
            step_minutes=15,
        )
