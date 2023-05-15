import datetime as dt

import pytest

from psp.dataset import DateSplits, auto_date_split

# Default split dates for our PV data source fixture.
D0 = dt.datetime(2020, 1, 1)
DM = dt.datetime(2020, 1, 16)
D1 = dt.datetime(2020, 1, 31)


@pytest.mark.parametrize(
    "num_trainings,expected_train_dates,expected_num_train,expected_num_test",
    [
        [1, [DM], 15, 15],
        [3, [DM, dt.datetime(2020, 1, 21), dt.datetime(2020, 1, 26)], 15, 15],
    ],
)
def test_auto_date_split(
    num_trainings, expected_train_dates, expected_num_train, expected_num_test
):
    splits = auto_date_split(D0, D1, num_trainings=num_trainings)
    print(splits)

    assert splits == DateSplits(
        train_dates=expected_train_dates,
        num_train_days=expected_num_train,
        num_test_days=expected_num_test,
    )
