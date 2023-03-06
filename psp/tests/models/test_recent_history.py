import datetime as dt

import numpy as np
import xarray as xr
from numpy.testing import assert_array_equal

from psp.models.recent_history import compute_history_per_horizon
from psp.typings import Horizons


def test_compute_history_per_horizon():
    raw_data = [
        #
        [dt.datetime(2000, 1, 1, 1), 2],
        [dt.datetime(2000, 1, 1, 5), 3],
        [dt.datetime(2000, 1, 1, 9), 4],
        [dt.datetime(2000, 1, 1, 11), 5],
        [dt.datetime(2000, 1, 1, 13), 6],
        #
        [dt.datetime(2000, 1, 2, 1), 5],
        [dt.datetime(2000, 1, 2, 4, 30), 12],
        [dt.datetime(2000, 1, 2, 5), 6],
        [dt.datetime(2000, 1, 2, 9), 7],
        [dt.datetime(2000, 1, 2, 10, 30), 13],
        #
        [dt.datetime(2000, 1, 3, 1), 8],
        [dt.datetime(2000, 1, 3, 5), 9],
        [dt.datetime(2000, 1, 3, 9), 10],
    ]

    dates, values = zip(*raw_data)

    array = xr.DataArray(list(values), coords={"ts": list(dates)}, dims=["ts"])

    now = dt.datetime(2000, 1, 3, 2, 30)

    # 7 to get one horizon in the second day
    horizons = Horizons(duration=4 * 60, num_horizons=7)

    history = compute_history_per_horizon(
        array,
        now=now,
        horizons=horizons,
    )

    # columns = day0 = 2000-1-1, day1= 2000-1-2
    # rows = horizon 0, horizon 1, ...
    expected_history = np.array(
        [
            # 2h30 - 6h30
            [np.nan, 3, 9],
            # 6h30 - 10h30
            [np.nan, 4, 7],
            # 10h30 - 14h30
            [np.nan, 5.5, 13],
            # 14h30 - 18h30
            [np.nan, np.nan, np.nan],
            # 18h30 - 22h30
            [np.nan, np.nan, np.nan],
            # 22h30 - 2h30
            [2, 5, 8],
            # 2h30 again
            [np.nan, 3, 9],
        ]
    )

    assert_array_equal(expected_history, history)
