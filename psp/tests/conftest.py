import pytest

from psp.data_sources.pv import NetcdfPvDataSource


@pytest.fixture
def pv_data_source():
    yield NetcdfPvDataSource(
        "psp/tests/fixtures/pv_data.netcdf",
        id_dim_name="ss_id",
        timestamp_dim_name="timestamp",
        rename={"generation_wh": "power"},
    )
