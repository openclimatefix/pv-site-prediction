import pandas as pd

from psp.data import C, remove_nights


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
