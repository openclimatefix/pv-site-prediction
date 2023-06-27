import datetime as dt

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

from psp.data_sources.nwp import NwpDataSource
from psp.models.recent_history import compute_history_per_horizon
from psp.serialization import load_model
from psp.testing import make_test_nwp_data, make_test_nwp_data_source
from psp.typings import Horizons, X


@pytest.fixture
def nwp_data_source():
    return make_test_nwp_data_source()


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


def _predict(pv_data_source, nwp_data_source):
    """Predict for given data sources.

    Common code for the test_predict_* tests.
    """
    model = load_model("psp/tests/fixtures/models/model_v8_2.pkl")

    model.set_data_sources(
        pv_data_source=pv_data_source,
        nwp_data_source=nwp_data_source,
    )

    return model.predict(X(ts=dt.datetime(2020, 1, 10, 12), pv_id="8215"))


def test_predict_with_missing_features(pv_data_source):
    nwp_data = make_test_nwp_data()
    nwp_data = nwp_data.sel(variable=["a", "b"])
    nwp_data_source = NwpDataSource(nwp_data)

    with pytest.raises(ValueError) as e:
        _predict(pv_data_source, nwp_data_source)
    assert "has 16 features" in str(e.value)
    assert "is expecting 18 features" in str(e.value)


def test_predict_with_extra_features(pv_data_source):
    nwp_data = make_test_nwp_data()

    # Add an extra variable
    var_d = nwp_data.sel(variable="a")
    var_d.coords["variable"] = "d"
    nwp_data = xr.concat([nwp_data, var_d], dim="variable")
    nwp_data_source = NwpDataSource(nwp_data)

    with pytest.raises(ValueError) as e:
        _predict(pv_data_source, nwp_data_source)
    assert "has 20 features" in str(e.value)
    assert "is expecting 18 features" in str(e.value)


def test_predict_with_features_in_wrong_order(pv_data_source):
    nwp_data = make_test_nwp_data()

    y1 = _predict(pv_data_source, NwpDataSource(nwp_data))

    # Swap the variables. The model should still use the right variables.
    nwp_data = nwp_data.assign_coords(variable=("variable", ["c", "a", "b"]))
    # FIXME
    nwp_data = nwp_data + 10

    nwp_data_source = NwpDataSource(nwp_data)

    y2 = _predict(pv_data_source, nwp_data_source)
    assert_allclose(y1.powers, y2.powers, atol=1e-6)
