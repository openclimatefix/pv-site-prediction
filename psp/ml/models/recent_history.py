import dataclasses
from datetime import datetime, timedelta
from typing import Iterator, Tuple

import numpy as np
import xarray as xr

from psp.data.data_sources.nwp import NwpDataSource
from psp.data.data_sources.pv import PvDataSource
from psp.ml.models.base import PvSiteModel, PvSiteModelConfig
from psp.ml.models.regressors.base import Regressor
from psp.ml.typings import Batch, Features, X, Y
from psp.pv import get_irradiance
from psp.utils.maths import safe_div


def to_midnight(ts: datetime) -> datetime:
    return ts.replace(hour=0, minute=0, second=0, microsecond=0)


def minutes_since_start_of_day(ts: datetime) -> float:
    """Time of day as minutes since midnight."""
    midnight = to_midnight(ts)
    return (ts - midnight).total_seconds() / 60.0


@dataclasses.dataclass
class SetupConfig:
    pv_data_source: PvDataSource
    nwp_data_source: NwpDataSource | None


class RecentHistoryModel(PvSiteModel):
    def __init__(
        self,
        config: PvSiteModelConfig,
        setup_config: SetupConfig,
        regressor: Regressor,
        use_nwp: bool = True,
        nwp_variables: list[str] | None = None,
    ):
        self._pv_data_source: PvDataSource
        self._nwp_data_source: NwpDataSource | None
        self._regressor = regressor
        self._use_nwp = use_nwp
        self._nwp_variables = nwp_variables

        self.setup(setup_config)

        if use_nwp:
            assert self._nwp_data_source is not None

        # We bump this when we make backward-incompatible changes in the code, to support old
        # serialized models.
        self._version = 1

        super().__init__(config, setup_config)

    def predict_from_features(self, features: Features) -> Y:
        pred = self._regressor.predict(features)
        powers = pred * features["factor"] * features["irradiance"]
        y = Y(powers=powers)
        return y

    def _get_time_of_day_stats(
        self,
        x: X,
        yesterday_midnight: datetime,
        future_interval: Tuple[float, float],
        data: xr.DataArray,
    ):

        start, end = future_interval

        pred_start = x.ts + timedelta(minutes=start)
        pred_start_minutes = minutes_since_start_of_day(pred_start)

        t0 = yesterday_midnight + timedelta(minutes=pred_start_minutes)
        t1 = t0 + timedelta(minutes=(end - start))

        # By default xarray ignores `nan` and returns `nan` for an empty list, which is
        # what we want.
        mean_power = data.sel(ts=slice(t0, t1)).mean().values

        if np.isnan(mean_power):
            return 0.0
        else:
            return mean_power

    def get_features(self, x: X) -> Features:
        features = dict()
        data_source = self._pv_data_source.without_future(
            x.ts, blackout=self.config.blackout
        )

        yesterday = x.ts - timedelta(days=1)
        yesterday_midnight = to_midnight(yesterday)

        # Slice as much as we can right away.
        data = data_source.get(
            pv_ids=x.pv_id,
            start_ts=yesterday_midnight,
            end_ts=x.ts,
        )["power"]

        coords = data.coords
        future_ts = [
            x.ts + timedelta(minutes=(f0 + f1) / 2)
            for f0, f1 in self.config.future_intervals
        ]

        lat = coords["latitude"].values
        lon = coords["longitude"].values

        # TODO `get_irradiance` returns a pandas dataframe. Maybe we could change that.
        irr = get_irradiance(
            lat=lat,
            lon=lon,
            # We add a timestamps for 15 minutes ago, because we'll do some stats on the last 30
            # minutes.
            timestamps=future_ts + [x.ts - timedelta(minutes=15)],
            tilt=coords["tilt"].values,
            orientation=coords["orientation"].values,
        )

        # TODO Should we use the other values from `get_irradiance` other than poa_global?
        irradiance = irr.loc[:, "poa_global"].to_numpy()

        irr_now = irradiance[-1]
        irradiance = irradiance[:-1]

        factor = coords["factor"].values
        features["irradiance"] = irradiance
        features["factor"] = factor

        # Get values at the same time of day as the prediction.

        yesterday_means = []
        for i, interval in enumerate(self._config.future_intervals):
            yesterday_mean = self._get_time_of_day_stats(
                x, yesterday_midnight, interval, data
            )
            yesterday_means.append(yesterday_mean)

        time_of_day_feats_arr: dict[str, np.ndarray] = {
            "yesterday_mean": safe_div(np.array(yesterday_means), irradiance * factor)
        }

        # Consider the NWP data in a small region around around our PV.
        if self._use_nwp:
            assert self._nwp_data_source is not None
            nwp_data = self._nwp_data_source.at(
                x.ts,
                nearest_lat=lat,
                nearest_lon=lon,
                load=True,
            )
            nwp_data_per_future = nwp_data.get(future_ts)
            for variable in (
                self._nwp_variables or self._nwp_data_source.list_variables()
            ):
                var_per_future = nwp_data_per_future.sel(variable=variable).values
                time_of_day_feats_arr[variable] = var_per_future

        # Concatenate all the per-future features in a matrix of dimensions (future, features)
        per_future = np.stack(list(time_of_day_feats_arr.values()), axis=-1)

        # Get the recent power
        recent_power = float(
            data.sel(ts=slice(x.ts - timedelta(minutes=30), x.ts)).mean()
        )
        is_nan = float(np.isnan(recent_power))
        if is_nan:
            recent_power = 0.0

        recent_power = safe_div(recent_power, irr_now * factor)

        features["per_future"] = per_future
        features["common"] = np.array([recent_power, is_nan])

        return features

    def train(
        self, train_iter: Iterator[Batch], valid_iter: Iterator[Batch], batch_size: int
    ):
        self._regressor.train(train_iter, valid_iter, batch_size)

    def setup(self, setup_config: SetupConfig):
        self._pv_data_source = setup_config.pv_data_source
        self._nwp_data_source = setup_config.nwp_data_source

    def get_state(self):
        state = self.__dict__.copy()
        # Do not save the data sources. Those should be set when loading the model using the `setup`
        # function.
        # We can't put that in __getstate__ directly because we need it when the model is pickled
        # for multiprocessing.
        del state["_pv_data_source"]
        del state["_nwp_data_source"]
        return state

    def set_state(self, state):
        if "_version" not in state:
            raise RuntimeError(
                "You are trying to load an older model with more recent code"
            )
        super().set_state(state)
