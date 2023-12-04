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

    # this is naughty, should not do this, should do this a _prepare_data function
    sat._data = sat._data.rename({"data": "value"})
    sat._data = sat._data.rename({"x_geostationary": "x", "y_geostationary": "y"})

    now = datetime(2021, 2, 1)
    sat.get(now=now, timestamps=[now])

    # need to translate from lat/long to geostationary, or the other way round

    # lat and long to geostaionary
    # import pyproj
    # area_definition_yaml = 'msg_seviri_rss_3km:\n  description: MSG SEVIRI Rapid Scanning Service area definition with 3 km resolution\n  projection:\n    proj: geos\n    lon_0: 9.5\n    h: 35785831\n    x_0: 0\n    y_0: 0\n    a: 6378169\n    rf: 295.488065897014\n    no_defs: null\n    type: crs\n  shape:\n    height: 298\n    width: 615\n  area_extent:\n    lower_left_xy: [28503.830075263977, 5090183.970808983]\n    upper_right_xy: [-1816744.1169023514, 4196063.827395439]\n    units: m\n' # noqa
    # import pyresample
    # geostationary_area_definition = pyresample.area_config.load_area_from_string(
    #     area_definition_yaml
    # )
    # geostationary_crs = geostationary_area_definition.crs
    #
    # lonlat_to_geostationary = pyproj.Transformer.from_crs(
    #     crs_from=4326,
    #     crs_to=geostationary_crs,
    #     always_xy=True,
    # ).transform
    #
    # x,y = lonlat_to_geostationary(xx=-1, yy=54)
    x = 5415727
    y = 1401188

    sat.get(now=now, timestamps=[now], max_lat=y + 1, min_lat=y, max_lon=x + 1, min_lon=x)
