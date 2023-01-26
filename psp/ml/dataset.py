import dataclasses
from datetime import datetime, timedelta
from typing import Iterator

import numpy as np
import pandas as pd
from torchdata.datapipes.iter import IterDataPipe

from psp.data.data_sources.pv import PvDataSource
from psp.ml.typings import FutureIntervals, PvId, Timestamp, X, Y


class PvXDataPipe(IterDataPipe[X]):
    """IterDataPipe that yields model inputs."""

    def __init__(
        self,
        data_source: PvDataSource,
        future_intervals: FutureIntervals,
        pv_ids: list[PvId] | None = None,
        start_ts: Timestamp | None = None,
        end_ts: Timestamp | None = None,
        step: int = 15,
    ):
        """
        Arguments:
        ---------
            pv_ids: If provided, will only pick from those pv_ids.
            start_ts: If provided, wll only pick dates after this date.
            end_ts: If provided, wll only pick dates before this date.
            step: Step used to make samples in time (in minutes).
        """
        self._data_source = data_source
        self._future_intervals = future_intervals
        self._pv_ids = pv_ids or self._data_source.list_pv_ids()
        self._start_ts = start_ts or self._data_source.min_ts()
        self._end_ts = end_ts or self._data_source.max_ts()
        self._step = step

        # Sanity checks.
        assert len(self._pv_ids) > 0
        assert self._end_ts > self._start_ts

    def __iter__(self) -> Iterator[X]:
        step = timedelta(minutes=self._step)

        for pv_id in self._pv_ids:
            ts = self._start_ts
            minute = ts.minute
            ts = ts.replace(
                minute=round_to(minute, self._step), second=0, microsecond=0
            )
            while ts < self._end_ts:
                x = X(pv_id=pv_id, ts=ts)
                yield x
                ts = ts + step


def round_to(x, to=1):
    return round(x / to) * to


# We inherit from PvSamplesGenerator to save some code even though it's not super sound.
class RandomPvXDataPipe(PvXDataPipe):
    """Infinite loop iterator of random PV data points."""

    def __init__(
        self,
        data_source: PvDataSource,
        future_intervals: FutureIntervals,
        random_state: np.random.RandomState,
        pv_ids: list[PvId] | None = None,
        start_ts: Timestamp | None = None,
        end_ts: Timestamp | None = None,
        step: int = 1,
    ):
        """
        Arguments
            step: Round the timestamp to this many minutes (with 0 seconds and 0 microseconds).
        """
        self._random_state = random_state
        super().__init__(data_source, future_intervals, pv_ids, start_ts, end_ts, step)

    def __iter__(self) -> Iterator[X]:

        num_seconds = (self._end_ts - self._start_ts).total_seconds()

        while True:
            # Random PV.
            pv_id = self._random_state.choice(self._pv_ids)

            # Random timestamp
            delta_seconds = self._random_state.random() * num_seconds
            ts = self._start_ts + timedelta(seconds=delta_seconds)

            # Round the minutes to a multiple of `steps`. This is particularly useful when testing,
            # where we might not want something as granualar as every minute, but want to be able
            # to aggregate many values for the *same* hour of day.
            minute = round_to(ts.minute, self._step)
            if minute > 59:
                minute = 0

            ts = ts.replace(minute=minute, second=0, microsecond=0)

            yield X(pv_id=pv_id, ts=ts)


def get_y_from_x(
    x: X, *, future_intervals: FutureIntervals, data_source: PvDataSource
) -> Y | None:
    """Given an input, compute the output.

    Return `None` if there is not output - it's simpler to filter those later.
    """
    min_future = min(i[0] for i in future_intervals)
    max_future = max(i[1] for i in future_intervals)
    data = data_source.get(
        x.pv_id,
        x.ts + timedelta(minutes=min_future),
        x.ts + timedelta(minutes=max_future),
    )["power"]

    if data.size == 0:
        return None

    # Find the targets for that pv/ts.
    # TODO Find a way to vectorize this.
    powers = []
    for start, end in future_intervals:
        ts0 = pd.Timestamp(x.ts + timedelta(minutes=start))
        ts1 = pd.Timestamp(x.ts + timedelta(minutes=end)) - timedelta(seconds=1)

        power_values = data.sel(ts=slice(ts0, ts1))

        if power_values.size == 0:
            powers.append(np.nan)
        else:
            power = float(power_values.mean())
            powers.append(power)

    powers_arr = np.array(powers)

    if np.all(np.isnan(powers_arr)):
        return None

    return Y(powers=powers_arr)


@dataclasses.dataclass
class DatasetSplit:
    """Dataset split on pv_ids and a time range."""

    start_ts: datetime
    end_ts: datetime
    pv_ids: list[PvId]

    def __repr__(self) -> str:
        return (
            f"<DataSplit pv_ids=<len={len(self.pv_ids)} start_ts={self.start_ts}"
            f" end_ts={self.end_ts}>"
        )


# A list of SS_ID that don't contain enough data.
# I just didn't want to calculate them everytime.
# TODO Get rid of those when we prepare the dataset.
SKIP_SS_IDS = set(
    [
        8440,
        16718,
        8715,
        17073,
        9108,
        9172,
        10167,
        10205,
        10207,
        10278,
        26778,
        26819,
        10437,
        10466,
        26915,
        10547,
        26939,
        26971,
        10685,
        10689,
        2638,
        2661,
        2754,
        2777,
        2783,
        2786,
        2793,
        2812,
        2829,
        2830,
        2867,
        2883,
        2904,
        2923,
        2947,
        2976,
        2989,
        2999,
        3003,
        3086,
        3118,
        3123,
        3125,
        3264,
        3266,
        3271,
        3313,
        3334,
        3470,
        3502,
        11769,
        11828,
        11962,
        3772,
        11983,
        3866,
        3869,
        4056,
        4067,
        4116,
        4117,
        4124,
        4323,
        4420,
        20857,
        4754,
        13387,
        13415,
        5755,
        5861,
        5990,
        6026,
        6038,
        6054,
        14455,
        6383,
        6430,
        6440,
        6478,
        6488,
        6541,
        6548,
        6560,
        14786,
        6630,
        6804,
        6849,
        6868,
        6870,
        6878,
        6901,
        6971,
        7055,
        7111,
        7124,
        7132,
        7143,
        7154,
        7155,
        7156,
        7158,
        7201,
        7237,
        7268,
        7289,
        7294,
        7311,
        7329,
        7339,
        7379,
        7392,
        7479,
        7638,
        7695,
        7772,
        15967,
        7890,
        16215,
        # This one has funny night values.
        7830,
    ]
)


@dataclasses.dataclass
class Splits:
    train: DatasetSplit
    valid: DatasetSplit
    test: DatasetSplit


def split_train_test(
    data_source: PvDataSource,
) -> Splits:
    # Note: Currently we hard-code a bunch of stuff in here, at some point we might want to make
    # some customizable.

    # Starting in 2020 because we only have NWP data from 2020.
    # TODO Get the NWP data for 2018 and 2019.
    # train_start = datetime(2018, 1, 1)
    train_start = datetime(2020, 1, 1)
    # Leaving a couple of days at the end to be safe.
    train_end = datetime(2020, 12, 29)

    test_start = datetime(2021, 1, 1)
    test_end = datetime(2022, 1, 1)

    pv_ids = set(data_source.list_pv_ids())
    pv_ids = pv_ids.difference(SKIP_SS_IDS)

    # Train on 90%.
    train_pv_ids = set(pv_id for pv_id in pv_ids if hash(pv_id) % 10 > 0)
    # Train on the remaining 10%.
    test_pv_ids = set(pv_id for pv_id in pv_ids if hash(pv_id) % 10 == 0)

    # We use the same time range for train and valid.
    # But we take some of the pv_ids.
    valid_pv_ids = set(pv_id for pv_id in train_pv_ids if hash(pv_id) % 13 == 1)

    # Remove those from the train set.
    train_pv_ids = train_pv_ids.difference(valid_pv_ids)

    assert len(train_pv_ids.intersection(valid_pv_ids)) == 0
    assert len(train_pv_ids.intersection(test_pv_ids)) == 0
    assert len(valid_pv_ids.intersection(test_pv_ids)) == 0

    return Splits(
        train=DatasetSplit(
            start_ts=train_start,
            end_ts=train_end,
            pv_ids=list(train_pv_ids),
        ),
        valid=DatasetSplit(
            start_ts=train_start,
            end_ts=train_end,
            pv_ids=list(valid_pv_ids),
        ),
        test=DatasetSplit(
            start_ts=test_start,
            end_ts=test_end,
            pv_ids=list(test_pv_ids),
        ),
    )
