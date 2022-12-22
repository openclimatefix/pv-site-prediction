from datetime import timedelta
from typing import TypedDict

import numpy as np
import xarray as xr

from psp.data.data_sources.pv import PvDataSource
from psp.data.uk_pv import C
from psp.ml.models.base import PvSiteModel
from psp.ml.typings import FutureIntervals, Timestamp, X, Y

BUFFER = timedelta(minutes=30)


class Features(TypedDict):
    yesterday_means: np.ndarray


class PreviousDayPvSiteModel(PvSiteModel[Features]):
    """Baseline that returns the power output of the previous day at the same time."""

    def __init__(self, *, data_source: PvDataSource, future_intervals: FutureIntervals):
        self._data_source = data_source

        super().__init__(future_intervals=future_intervals)

    def _get_features_for_one_ts(self, data: xr.DataArray, ts: Timestamp) -> float:
        start = ts - BUFFER
        end = ts + BUFFER

        da = data.sel({C.date: slice(start, end)})

        if da.size == 0:
            return np.nan
        else:
            return float(da.mean())

    def _predict_from_features(self, x: X, features: Features) -> Y:
        powers = features["yesterday_means"]
        return Y(powers=powers)

    def get_features(self, x: X) -> Features:
        max_ts = max(x[1] for x in self.future_intervals)

        yesterday = x.ts - timedelta(days=1)

        start = yesterday - BUFFER
        end = yesterday + timedelta(minutes=max_ts) + BUFFER

        data = self._data_source.get(
            pv_ids=x.pv_id,
            start_ts=start,
            end_ts=end,
        )[C.power]

        powers = [
            self._get_features_for_one_ts(
                data, yesterday + timedelta(minutes=(start + end) // 2)
            )
            for [start, end] in self.future_intervals
        ]

        return dict(
            yesterday_means=np.array(powers),
        )
