import dataclasses
from datetime import timedelta

import numpy as np
import xarray as xr

from psp.data.data_sources.pv import PvDataSource
from psp.ml.models.base import PvSiteModel, PvSiteModelConfig
from psp.ml.typings import Features, Timestamp, X, Y

BUFFER = timedelta(minutes=30)


@dataclasses.dataclass
class SetupConfig:
    data_source: PvDataSource


class YesterdayPvSiteModel(PvSiteModel):
    """Baseline that returns the power output of the previous day at the same time."""

    def __init__(self, config: PvSiteModelConfig, setup_config: SetupConfig):
        self._data_source = setup_config.data_source
        super().__init__(config, setup_config)

    def predict_from_features(self, features: Features) -> Y:
        powers = features["yesterday_means"]
        assert isinstance(powers, np.ndarray)
        return Y(powers=powers)

    def get_features(self, x: X) -> Features:
        data_source = self._data_source.without_future(
            x.ts, blackout=self.config.blackout
        )
        max_minutes = max(x[1] for x in self.config.future_intervals)

        yesterday = x.ts - timedelta(days=1)

        start = yesterday - BUFFER
        end = yesterday + timedelta(minutes=max_minutes) + BUFFER

        data = data_source.get(
            pv_ids=x.pv_id,
            start_ts=start,
            end_ts=end,
        )["power"]

        powers = [
            self._get_features_for_one_ts(
                data, yesterday + timedelta(minutes=(start + end) // 2)
            )
            for [start, end] in self.config.future_intervals
        ]

        return dict(
            yesterday_means=np.array(powers),
        )

    def _get_features_for_one_ts(self, data: xr.DataArray, ts: Timestamp) -> float:
        start = ts - BUFFER
        end = ts + BUFFER

        da = data.sel(ts=slice(start, end))

        if da.size == 0:
            return np.nan
        else:
            return float(da.mean())
