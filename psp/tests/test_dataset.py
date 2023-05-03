import datetime as dt

import pytest

from psp.dataset import DatasetSplit, split_train_test

# Default split dates for our PV data source fixture.
D0 = dt.datetime(2020, 1, 1)
D1 = dt.datetime(2020, 1, 6)
D2 = dt.datetime(2020, 1, 7)
D3 = dt.datetime(2020, 1, 13)


@pytest.mark.parametrize(
    "train_start,train_end,test_start,test_end",
    [
        # All of those should be equivalent and cover the most common use-cases.
        [None, None, None, None],
        [D0, None, None, None],
        [None, None, None, D3],
        [D0, None, None, D3],
        [D0, D1, None, D3],
        [D0, None, D2, D3],
        [D0, D1, None, None],
    ],
)
def test_split_train_test_default(pv_data_source, train_start, train_end, test_start, test_end):
    splits = split_train_test(
        pv_data_source,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )

    assert splits.train == DatasetSplit(
        start_ts=D0,
        end_ts=D1,
        pv_ids=["8215", "8229"],
    )

    assert splits.valid == DatasetSplit(
        start_ts=D0,
        end_ts=D1,
        # We would need a more interesting data source to have non-trivial valid and test splits.
        pv_ids=[],
    )

    assert splits.test == DatasetSplit(
        start_ts=D2,
        end_ts=D3,
        pv_ids=[],
    )
