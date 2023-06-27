import pytest

from psp.data_sources.pv import NetcdfPvDataSource
from psp.testing import make_test_nwp_data


@pytest.fixture
def pv_data_source():
    yield NetcdfPvDataSource(
        "psp/tests/fixtures/pv_data.netcdf",
        id_dim_name="ss_id",
        timestamp_dim_name="timestamp",
        rename={"generation_wh": "power"},
    )

@pytest.fixture
def nwp_data():
    pass

@pytest.fixture
def nwp_data_source(pv_data_source):
    # Create data that aligns with the pv_data_source defined in the same file.
    time = [dt.datetime(2020, 1, 1) + dt.timedelta(hours=6) * i for i in range(14 * 4)]
    x = [0, 0.5, 1, 1.5, 2]
    y = [0, 0.5, 1, 1.5, 2]
    step = [dt.timedelta(hours=x) for x in [0, 3, 6, 9, 12]]
    variable = ["a", "b", "c"]

    data = np.zeros(
        (
            len(time),
            len(step),
            len(x),
            len(y),
            len(variable),
        )
    )

    da = xr.DataArray(data, dict(time=time, step=step, x=x, y=y, variable=variable))

    ds = xr.Dataset(dict(value=da))

    return ds
