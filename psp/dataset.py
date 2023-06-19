import dataclasses
from datetime import datetime, timedelta

from psp.data_sources.pv import PvDataSource
from psp.typings import PvId
from psp.utils.hashing import naive_hash


@dataclasses.dataclass
class PvSplits:
    train: list[PvId]
    valid: list[PvId]
    test: list[PvId]


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
