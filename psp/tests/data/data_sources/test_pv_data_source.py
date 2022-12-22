from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from psp.data.data_sources.pv import NetcdfPvDataSource


@pytest.fixture
def pv_data_source(tmp_path):
    pv_ids = [1, 2, 3]
    t0 = datetime(2023, 1, 1)
    dt = timedelta(days=1)
    ts_range = [t0 + i * dt for i in range(4)]

    n = len(pv_ids)
    m = len(ts_range)

    # (pv_id, timestamp)
    data = np.arange(n * m, dtype=float).reshape(n, m)

    da = xr.DataArray(
        data=data,
        coords={
            # Using the same column names as in the original dataset.
            "ss_id": pv_ids,
            "timestamp": ts_range,
        },
        dims=["ss_id", "timestamp"],
    )

    path = tmp_path / "pv_data_source.netcdf"
    d = xr.Dataset({"generation_wh": da})
    d.to_netcdf(path)
    return NetcdfPvDataSource(path)


def test_pv_data_source_ignore_future(pv_data_source):
    # Without `ignore_future`.
    assert pv_data_source.min_ts() == datetime(2023, 1, 1)
    assert pv_data_source.max_ts() == datetime(2023, 1, 4)
    assert pv_data_source.list_pv_ids() == [1, 2, 3]
    assert (
        pv_data_source.get(
            pv_ids=1, start_ts=datetime(2023, 1, 2), end_ts=datetime(2023, 1, 3)
        )["power"].size
        == 2
    )

    # With `ignore_future`.
    new_data_source = pv_data_source.without_future(datetime(2023, 1, 3))
    assert new_data_source.min_ts() == datetime(2023, 1, 1)
    assert new_data_source.max_ts() == datetime(2023, 1, 2, 23, 59, 59)
    assert (
        new_data_source.get(
            pv_ids=1, start_ts=datetime(2023, 1, 2), end_ts=datetime(2023, 1, 3)
        )["power"].size
        == 1
    )

    # After the decorator we are back to normal.
    assert (
        pv_data_source.get(
            pv_ids=1, start_ts=datetime(2023, 1, 2), end_ts=datetime(2023, 1, 3)
        )["power"].size
        == 2
    )
