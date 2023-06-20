"""Utils used in tests."""

import datetime as dt

import numpy as np
import xarray as xr
from click.testing import CliRunner

from psp.data_sources.nwp import NwpDataSource


def run_click_command(main_func, cmd_args: list[str]):
    """Run a click command in a test-fiendly way."""
    runner = CliRunner()

    result = runner.invoke(main_func, cmd_args, catch_exceptions=True)

    # Without this the output to stdout/stderr is grabbed by click's test runner.
    print(result.output)

    # In case of an exception, raise it so that the test fails with the exception.
    if result.exception:
        raise result.exception

    assert result.exit_code == 0

    return result


def make_test_nwp_data() -> xr.Dataset:
    rng = np.random.default_rng(seed=1234)
    # Create data that aligns with the pv_data_source defined in the same file.
    time = [dt.datetime(2020, 1, 1) + dt.timedelta(hours=6) * i for i in range(14 * 4)]
    x = [0, 0.5, 1, 1.5, 2]
    y = [0, 0.5, 1, 1.5, 2]
    step = [dt.timedelta(hours=x) for x in [0, 3, 6, 9, 12]]
    variable = ["a", "b", "c"]

    data = rng.uniform(
        size=(
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


def make_test_nwp_data_source():
    """Make fake NWP data with dates and locations that matches our fixture PV data."""
    ds = make_test_nwp_data()
    return NwpDataSource(ds)
