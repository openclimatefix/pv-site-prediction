import numpy as np
import pytest
from numpy.testing import assert_allclose

from psp.gis import approx_add_meters_to_lat_lon


@pytest.mark.parametrize(
    "p0",
    [
        np.array([[50, 0], [45.543526844263546, -73.5694752214446]]),
        np.array([1, 1]),
        [1, 1],
    ],
)
def test_approx_add_meters_to_lat_lon_happy_path(p0):
    p1 = approx_add_meters_to_lat_lon(p0, [0, 10])
    p2 = approx_add_meters_to_lat_lon(p1, [-10, 0])
    p3 = approx_add_meters_to_lat_lon(p2, [10, -10])

    assert_allclose(p0, p3, atol=1e-10)

    # Weird but should also work!
    p1 = approx_add_meters_to_lat_lon(p0, p0)
    p2 = approx_add_meters_to_lat_lon(p1, -np.array(p0))

    assert_allclose(p0, p2, atol=1e-10)
