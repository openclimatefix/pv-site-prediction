from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from psp.data.data_sources.pv import NetcdfPvDataSource


def _make_pv_data_xarray() -> xr.Dataset:
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
            "pv_id": pv_ids,
            "ts": ts_range,
        },
        dims=["pv_id", "ts"],
    )
    d = xr.Dataset({"power": da})
    return d


@pytest.fixture
def pv_data_source(tmp_path):
    d = _make_pv_data_xarray()
    path = tmp_path / "pv_data_source.netcdf"
    d.to_netcdf(path)
    return NetcdfPvDataSource(
        path,
    )


def test_pv_data_source_rename(tmp_path):
    d = _make_pv_data_xarray()
    d = d.rename({"pv_id": "mon_id", "ts": "mon_ts", "power": "mon_power"})
    path = tmp_path / "pv_data_source_wrong_col.netcdf"
    d.to_netcdf(path)

    ds = NetcdfPvDataSource(
        path,
        id_dim_name="mon_id",
        timestamp_dim_name="mon_ts",
        rename={"mon_power": "power"},
    )

    assert (
        ds.get(pv_ids="1", start_ts=datetime(2023, 1, 2), end_ts=datetime(2023, 1, 3))["power"].size
        == 2
    )


def test_pv_data_source_ignore_future(pv_data_source):
    # Without `ignore_future`.
    assert pv_data_source.min_ts() == datetime(2023, 1, 1)
    assert pv_data_source.max_ts() == datetime(2023, 1, 4)
    assert pv_data_source.list_pv_ids() == "1 2 3".split()
    assert (
        pv_data_source.get(pv_ids="1", start_ts=datetime(2023, 1, 2), end_ts=datetime(2023, 1, 3))[
            "power"
        ].size
        == 2
    )

    # With `ignore_future`.
    new_data_source = pv_data_source.without_future(datetime(2023, 1, 3))
    assert new_data_source.min_ts() == datetime(2023, 1, 1)
    assert new_data_source.max_ts() == datetime(2023, 1, 2, 23, 59, 59)
    assert (
        new_data_source.get(pv_ids="1", start_ts=datetime(2023, 1, 2), end_ts=datetime(2023, 1, 3))[
            "power"
        ].size
        == 1
    )

    # After the decorator we are back to normal.
    assert (
        pv_data_source.get(pv_ids="1", start_ts=datetime(2023, 1, 2), end_ts=datetime(2023, 1, 3))[
            "power"
        ].size
        == 2
    )


def test_ignore_pv_ids(tmp_path):
    d = _make_pv_data_xarray()
    path = tmp_path / "pv_data_source_test_ignore_pv.netcdf"
    d.to_netcdf(path)

    ds = NetcdfPvDataSource(path, ignore_pv_ids=["2", "3"])

    assert ds.list_pv_ids() == ["1"]

    with pytest.raises(KeyError):
        ds.get(pv_ids=["2"])

    with pytest.raises(KeyError):
        ds.get(pv_ids=["1", "3"])
