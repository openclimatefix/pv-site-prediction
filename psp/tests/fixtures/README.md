## NWP data

The nwp data has been generated with:

```python
import datetime as dt

import ocf_blosc2
import xarray as xr

from psp.data_sources.nwp import _slice_on_lat_lon
from psp.gis import CoordinateTransformer

nwp = xr.open_dataset(
    "/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP"
    "/UK_Met_Office/UKV/zarr/UKV_2020_NWP.zarr"
)

nwp = _slice_on_lat_lon(
    nwp,
    min_lat=51.1,
    max_lat=53.7,
    min_lon=-3.3,
    max_lon=-2.7,
    transformer=CoordinateTransformer(4326, 27700),
    x_is_ascending=True,
    y_is_ascending=False,
)

nwp = nwp.sel(variable=["dswrf", "lcc", "mcc", "hcc"])

nwp = nwp.sel(init_time=slice(dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 14)))

nwp = nwp.isel(x=[0, -1], y=[0, -1])

nwp = nwp.sel(step=slice(dt.timedelta(hours=0), dt.timedelta(hours=4)))

for k in list(nwp['UKV'].attrs):
    del nwp['UKV'].attrs[k]

chunks = {
    "init_time": -1,
    "x": -1,
    "y": -1,
    "step": -1,
    "variable": -1,
}

nwp.to_zarr(
    "psp/tests/fixtures/nwp.zarr",
    mode="w",
    encoding={"UKV": {"chunks": [chunks[d] for d in nwp.dims]}},
)
```
