import pytest

from psp.data.data_sources.pv import NetcdfPvDataSource


@pytest.fixture
def pv_data_source():
    yield NetcdfPvDataSource("psp/tests/fixtures/pv_data.netcdf")
