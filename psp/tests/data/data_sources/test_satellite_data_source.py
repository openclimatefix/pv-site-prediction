import xarray as xr

from psp.data_sources.satellite import SatelliteDataSource


def test_satellite_data_source():
    """Test loading the satellite data

    Note this test uses the satellite public dataset to get this data.

    """
    paths = [
        "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v4/2021_nonhrv.zarr"
    ]

    xr.open_zarr(
        "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v4/2022_nonhrv.zarr",
        consolidated=True,
    )

    _ = SatelliteDataSource(paths_or_data=paths)
