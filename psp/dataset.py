import dataclasses
from datetime import datetime, timedelta
from typing import Iterator

import numpy as np
import pandas as pd
from torchdata.datapipes.iter import IterDataPipe

from psp.data_sources.pv import PvDataSource
from psp.typings import Horizons, PvId, Timestamp, X, Y
from psp.utils.hashing import naive_hash


class PvXDataPipe(IterDataPipe[X]):
    """IterDataPipe that yields model inputs."""

    def __init__(
        self,
        data_source: PvDataSource,
        horizons: Horizons,
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
        self._horizons = horizons
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
            ts = ts.replace(minute=round_to(minute, self._step), second=0, microsecond=0)
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
        horizons: Horizons,
        random_state: np.random.RandomState,
        pv_ids: list[PvId] | None = None,
        start_ts: Timestamp | None = None,
        end_ts: Timestamp | None = None,
        step: int = 1,
    ):
        """
        Arguments:
        ---------
            step: Round the timestamp to this many minutes (with 0 seconds and 0 microseconds).
        """
        self._random_state = random_state
        super().__init__(data_source, horizons, pv_ids, start_ts, end_ts, step)

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


def get_y_from_x(x: X, *, horizons: Horizons, data_source: PvDataSource) -> Y | None:
    """Given an input, compute the output.

    Return `None` if there is not output - it's simpler to filter those later.
    """
    min_horizon = min(i[0] for i in horizons)
    max_horizon = max(i[1] for i in horizons)
    data = data_source.get(
        x.pv_id,
        x.ts + timedelta(minutes=min_horizon),
        x.ts + timedelta(minutes=max_horizon),
    )["power"]

    if data.size == 0:
        return None

    # Find the targets for that pv/ts.
    # TODO Find a way to vectorize this.
    powers = []
    for start, end in horizons:
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
class PvSplits:
    train: list[PvId]
    valid: list[PvId]
    test: list[PvId]


def pv_list_to_short_str(x: list[PvId]) -> str:
    """Util to format a list of PV ids into a small string."""
    if len(x) < 4:
        return str(x)
    else:
        return f"[{repr(x[0])}, {repr(x[1])}, ..., {repr(x[-1])}]"


def _floor_date(date: datetime) -> datetime:
    """Round date down to midnight."""
    return datetime(*date.timetuple()[:3])


def _ceiling_date(date: datetime) -> datetime:
    """Round date up to next midnight"""
    rounded = _floor_date(date)
    if rounded == date:
        return date
    else:
        return date + timedelta(days=1)


def split_pvs(
    pv_data_source: PvDataSource,
    *,
    pv_split: float | None = 0.9,
    valid_split: float = 0.1,
) -> PvSplits:
    """
    Split the PV ids in a PV Data Source into train/valid/test sets.

    Arguments:
    ---------
        pv_split: Ratio of PV sites to put in the train set. The rest will go in the test set. Use
            an explicit `None` to *not* split on PV ids (use all the PV ids for both train and
            test). This can make sense in use-cases where there is a small and stable number PV
            sites.
        valid_split: Ratio of Pv sites from the train set to use as valid set. Note that the
            time range is the same for train and valid.
    """
    pv_ids = set(pv_data_source.list_pv_ids())

    if pv_split is None:
        train_pv_ids = pv_ids
        valid_pv_ids = pv_ids
        test_pv_ids = pv_ids
    else:
        assert isinstance(pv_split, float)
        # We split on a hash of the pv_ids.
        train_pv_ids = set(
            pv_id for pv_id in pv_ids if ((naive_hash(pv_id) % 1000) < 1000 * pv_split)
        )
        test_pv_ids = set(
            pv_id for pv_id in pv_ids if ((naive_hash(pv_id) % 1000) >= 1000 * pv_split)
        )

        # We use the same time range for train and valid.
        # But we take some of the pv_ids, using the same kind of heuristic as the train/tests split.
        valid_pv_ids = set(
            pv_id
            for pv_id in train_pv_ids
            if ((naive_hash(pv_id + " - hack to get a different hash") % 1000) < 1000 * valid_split)
        )

        # Remove those from the train set.
        train_pv_ids = train_pv_ids.difference(valid_pv_ids)

        assert len(train_pv_ids.intersection(valid_pv_ids)) == 0
        assert len(train_pv_ids.intersection(test_pv_ids)) == 0
        assert len(valid_pv_ids.intersection(test_pv_ids)) == 0

    # Note the `sorted`. This is because `set` can mess up the order and we want the randomness we
    # will add later (when picking pv_ids at random) to be deterministic.
    return PvSplits(
        train=list(sorted(train_pv_ids)),
        valid=list(sorted(valid_pv_ids)),
        test=list(sorted(test_pv_ids)),
    )


@dataclasses.dataclass
class TrainDateSplit:
    """Train part of the `DateSplits` data class.

    train_date: The date at which we are training.
    train_days: The maximum number of days to use, prior to `train_date`.
    """

    train_date: datetime
    train_days: int


@dataclasses.dataclass
class TestDateSplit:
    """Test part of the `DateSplits` data class.

    start_date: Start date of the test set.
    end_date: End date of the test set.
    """

    start_date: datetime
    end_date: datetime


@dataclasses.dataclass
class DateSplits:
    """Defines the train/test scheme for training and evaluating."""

    train_date_splits: list[TrainDateSplit]
    test_date_split: TestDateSplit


def auto_date_split(
    test_start_date: datetime,
    test_end_date: datetime,
    *,
    train_days: int,
    num_trainings: int = 1,
) -> DateSplits:
    """Make a `DateSplits` that tests on a given time range and trains
    `num_trainings` times evenly spaced.
    """
    train_splits: list[TrainDateSplit] = []

    d0 = test_start_date - timedelta(days=1)
    num_days_test = (test_end_date - d0).days

    train_splits = [
        TrainDateSplit(
            train_date=d0 + timedelta(days=i * num_days_test // num_trainings),
            train_days=train_days,
        )
        for i in range(num_trainings)
    ]

    return DateSplits(
        train_date_splits=train_splits,
        test_date_split=TestDateSplit(
            start_date=test_start_date,
            end_date=test_end_date,
        ),
    )
