import dataclasses
from datetime import datetime, timedelta
from typing import Iterator, Tuple

import numpy as np
import xarray as xr

from psp.ml.dataset import PvDataSource
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
    data_source: PvDataSource


class RecentHistoryModel(PvSiteModel):
    def __init__(
        self,
        config: PvSiteModelConfig,
        setup_config: SetupConfig,
        regressor: Regressor,
    ):
        self._data_source: PvDataSource
        self.setup(setup_config)
        self._regressor = regressor
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
        data_source = self._data_source.without_future(
            x.ts, blackout=self.config.blackout
        )

        # TODO Compare with and without the `load()`.
        data = data_source.get(
            pv_ids=x.pv_id,
            start_ts=x.ts - timedelta(days=30),
            end_ts=x.ts,
        )["power"].load()

        coords = data.coords
        future_ts = [
            x.ts + timedelta(minutes=(f0 + f1) / 2)
            for f0, f1 in self.config.future_intervals
        ]

        # TODO `get_irradiance` returns a pandas dataframe. Maybe we could change that.
        irr = get_irradiance(
            lat=coords["latitude"].values,
            lon=coords["longitude"].values,
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

        yesterday = x.ts - timedelta(days=1)
        yesterday_midnight = to_midnight(yesterday)

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
        self._data_source = setup_config.data_source
