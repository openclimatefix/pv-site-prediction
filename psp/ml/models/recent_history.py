import dataclasses
from collections import defaultdict
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


AGGS = ["nanmean", "nanmedian", "nanstd", "nanmin", "nanmax"]


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
        y = Y(powers=pred * features["factor"] * features["irradiance"])
        return y

    def _get_time_of_day_stats(
        self,
        x: X,
        yesterday_midnight: datetime,
        future_interval: Tuple[float, float],
        data: xr.DataArray,
    ):

        start, end = future_interval
        mean_interval = (start + end) / 2
        radius = (end - start) / 2

        ts_pred = x.ts + timedelta(minutes=mean_interval)
        pred_time_of_day = minutes_since_start_of_day(ts_pred)

        num_days = 10

        previous_days_ts = [
            (
                # TODO clean that ugly radius calculation
                yesterday_midnight
                - timedelta(days=i)
                + timedelta(minutes=pred_time_of_day - radius),
                yesterday_midnight
                - timedelta(days=i)
                + timedelta(minutes=pred_time_of_day + radius),
            )
            for i in range(num_days)
        ]

        # TODO vectorize this query?
        # TODO check how to do those groupby - our use-case isn't supported
        # data.groupby_bins('timestamp', previous_days_ts)
        powers = [
            # TODO what about the boundaries (we don't want the end often)
            # TODO
            # Or we could just get proxy object in here, like do something like
            # sample_data =
            # self.data_source.get(pv_ids=x.pv_id, start_ts=ts - 2weeks, ts=ts).compute()
            # and then refer to that. We should do it and compare the speed
            # TODO Also check for the time difference when using a machine in the cluster, not sure
            # how reading from
            # Google Storage is slower than reading from my laptop.
            # TODO maybe don't do a mean per day, maybe do it later.
            data.sel(ts=slice(start, end)).mean()
            for start, end in previous_days_ts
        ]

        # TODO normalize the power, etc.

        is_all_nan = np.all(np.isnan(powers))
        if is_all_nan:
            aggs = {name: 0.0 for name in AGGS}
        else:
            aggs = {name: getattr(np, name)(powers) for name in AGGS}

        return aggs

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

        # TODO Maybe everything can be vectorized.
        time_of_day_feats = defaultdict(list)
        for i, interval in enumerate(self._config.future_intervals):
            aggs = self._get_time_of_day_stats(x, yesterday_midnight, interval, data)
            for agg_name, value in aggs.items():
                time_of_day_feats["lastweek_" + agg_name].append(value)

        time_of_day_feats_arr: dict[str, np.ndarray] = dict()
        for key, value in time_of_day_feats.items():
            time_of_day_feats_arr[key] = safe_div(value, irradiance * factor)

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
