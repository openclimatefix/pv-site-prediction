import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from psp.data import C, get_max_power_for_time_of_day, remove_nights


def test_remove_nights():

    night1 = pd.Timestamp("2022-1-1 1:00", tz="utc")
    night2 = pd.Timestamp("2022-1-1 2:00", tz="utc")
    day1 = pd.Timestamp("2022-1-1 8:00", tz="utc")
    day2 = pd.Timestamp("2022-1-1 12:00", tz="utc")
    day3 = pd.Timestamp("2022-1-1 15:00", tz="utc")
    night3 = pd.Timestamp("2022-1-1 21:00", tz="utc")

    df = pd.DataFrame(
        {
            C.POWER: [0, 0, 1, 2, 3, 4, 4, 5],
            C.DATE: [
                night1,
                night1,
                night2,
                day1,
                day1,
                day2,
                day3,
                night3,
            ],
            C.ID: [0, 1, 0, 0, 1, 0, 0, 0],
        }
    )

    meta = pd.DataFrame(
        {
            C.ID: [0, 1, 2, 3, 4, 5],
            C.LAT: [0, 1, 1, 2, 3, 4],
            C.LON: [0, 4, 1, 2, 3, 4],
        }
    )

    df2 = remove_nights(df, meta)

    assert len(df2) == 4
    assert df2[C.DATE].tolist() == [day1, day1, day2, day3]


def _ts(d, h):
    return pd.Timestamp(year=2022, month=1, day=d, hour=h)


def _from_records(rec):
    return pd.DataFrame.from_records(rec, columns=[C.id, C.date, C.power]).set_index(
        [C.id, C.date]
    )


def test_get_max_power_for_time_of_day():

    df = _from_records(
        [
            [1, _ts(1, 1), 1.0],
            [1, _ts(1, 2), 3],
            [1, _ts(2, 1), 5],
            [1, _ts(2, 2), 7],
            [1, _ts(3, 1), 9],
            [1, _ts(4, 1), 10],
            [2, _ts(1, 1), 4],
            [2, _ts(1, 2), 20],
            [2, _ts(2, 1), 30],
        ],
    )

    max_0 = get_max_power_for_time_of_day(df, radius=0)

    assert_frame_equal(max_0, df)

    max_1 = get_max_power_for_time_of_day(df, radius=1)

    expected = _from_records(
        [
            [1, _ts(1, 1), 5.0],
            [1, _ts(1, 2), 7],
            [1, _ts(2, 1), 9],
            [1, _ts(2, 2), 7],
            [1, _ts(3, 1), 10],
            [1, _ts(4, 1), 10],
            [2, _ts(1, 1), 30],
            [2, _ts(1, 2), 20],
            [2, _ts(2, 1), 30],
        ]
    )

    assert_frame_equal(max_1, expected)

    max_2 = get_max_power_for_time_of_day(df, radius=2)

    expected = _from_records(
        [
            [1, _ts(1, 1), 9.0],
            [1, _ts(1, 2), 7],
            [1, _ts(2, 1), 10],
            [1, _ts(2, 2), 7],
            [1, _ts(3, 1), 10],
            [1, _ts(4, 1), 10],
            [2, _ts(1, 1), 30],
            [2, _ts(1, 2), 20],
            [2, _ts(2, 1), 30],
        ],
    )

    assert_frame_equal(max_2, expected)

    # Test the `min_records` parameter.
    max_2_records = get_max_power_for_time_of_day(df, radius=2, min_records=2)

    expected = _from_records(
        [
            [1, _ts(1, 1), 9.0],
            [1, _ts(1, 2), 7],
            [1, _ts(2, 1), 10],
            [1, _ts(2, 2), 7],
            [1, _ts(3, 1), 10],
            [1, _ts(4, 1), 10],
            [2, _ts(1, 1), 30],
            [2, _ts(1, 2), np.nan],
            [2, _ts(2, 1), 30],
        ],
    )

    assert_frame_equal(max_2_records, expected)
