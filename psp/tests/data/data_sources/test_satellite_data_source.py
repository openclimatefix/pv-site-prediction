from datetime import datetime

import ocf_blosc2  # noqa

from psp.data_sources.satellite import SatelliteDataSource


def test_satellite_data_source():
    """Test loading the satellite data

    Note this test uses the satellite public dataset to get this data.

    """
    # this is wrong the google datasets
    paths = [
        "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v4/2021_nonhrv.zarr"
    ]

    sat = SatelliteDataSource(paths_or_data=paths, x_is_ascending=False)

    now = datetime(2021, 2, 1)
    # sat.get(now=now, timestamps=[now])

    x = 5415727
    y = 1401188

    x, y = sat.lonlat_to_geostationary(xx=-1, yy=54)

    assert x > sat._data.x.min()
    assert x < sat._data.x.max()
    assert y > sat._data.y.min()
    assert y < sat._data.y.max()

    example = sat.get(now=now, timestamps=[now], nearest_lat=y, nearest_lon=x)

    example = sat.get(
        now=now, timestamps=[now], max_lat=y + 20000, min_lat=y, max_lon=x + 10000, min_lon=x
    )
    assert example.x.size > 0
    assert example.y.size > 0
