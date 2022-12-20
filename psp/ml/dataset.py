import datetime
from typing import Iterator

import numpy as np
import pandas as pd
from torchdata.datapipes.iter import IterDataPipe

from psp.data.data_sources.pv import PvDataSource
from psp.data.uk_pv import C
from psp.ml.types import FutureIntervals, PvId, Timestamp, X, Y


# TODO Do a random version for training - or make this one optionally random?
class PvXDataPipe(IterDataPipe[X]):
    """IterDataPipe that yields model inputs."""

    def __init__(
        self,
        data_source: PvDataSource,
        future_intervals: FutureIntervals,
        # random_state: np.random.RandomState,
        # If provided, will only pick from those pv_ids.
        pv_ids: list[PvId] | None = None,
        # If provided, wll only pick dates after this date.
        min_ts: Timestamp | None = None,
        max_ts: Timestamp | None = None,
    ):
        self._data_source = data_source
        # self._rnd = random_state
        self._future_intervals = future_intervals
        self._pv_ids = pv_ids or self._data_source.list_pv_ids()
        self._min_ts = min_ts or self._data_source.min_ts()
        self._max_ts = max_ts or self._data_source.max_ts()

        # Sanity checks.
        assert len(self._pv_ids) > 0
        assert self._max_ts > self._min_ts

    def __iter__(self) -> Iterator[X]:
        # TODO This should be a parameter of the constructor.
        step_minutes = 15

        num_steps = int((self._max_ts - self._min_ts).total_seconds() / 60 / 15)

        assert num_steps > 0

        for pv_id in self._pv_ids:
            for step in range(num_steps):
                ts = self._min_ts + datetime.timedelta(minutes=step * step_minutes)
                x = X(pv_id=pv_id, ts=ts)
                yield x


# class RandomPvSamplesGenerator(PvSamplesGenerator):
#     def _iter_x(self) -> Iterator[X]:
#         raise NotImplementedError
#         min_future = min(i[0] for i in self._future_intervals)
#         max_future = max(i[1] for i in self._future_intervals)

# One "step" per minute.
#         num_steps = int(
#             (max_ts - min_ts - datetime.timedelta(minutes=max_future)).total_seconds()
#             / 60
#         )

#         while True:
# Random PV.
#             pv_id = self._rnd.choice(pv_ids)
#             step = self._rnd.randint(0, num_steps)

# Random timestamp: this is the time at which we are making the prediction.
#             ts = min_ts + datetime.timedelta(minutes=step)
#             yield X(pv_id=pv_id, ts=ts)


def get_y_from_x(
    x: X, *, future_intervals: FutureIntervals, data_source: PvDataSource
) -> Y | None:
    """Given an input, compute the output.

    Return `None` if there is not output.
    """
    min_future = min(i[0] for i in future_intervals)
    max_future = max(i[1] for i in future_intervals)
    data = data_source.get(
        x.pv_id,
        x.ts + datetime.timedelta(minutes=min_future),
        x.ts + datetime.timedelta(minutes=max_future),
    )[C.power]

    if data.size == 0:
        return None

    # Find the targets for that pv/ts.
    powers = []
    for start, end in future_intervals:
        # Again `str` because of a bug in pandas.
        ts0 = pd.Timestamp(x.ts + datetime.timedelta(minutes=start))
        ts1 = pd.Timestamp(x.ts + datetime.timedelta(minutes=end)) - datetime.timedelta(
            seconds=1
        )

        power_values = data.sel({C.id: x.pv_id, C.date: slice(ts0, ts1)})

        if power_values.size == 0:
            powers.append(np.nan)
        else:
            power = float(power_values.mean())
            powers.append(power)

    return Y(powers=np.array(powers))
