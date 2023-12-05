import pyproj
import pyresample
import xarray as xr

from psp.data_sources.nwp import NwpDataSource
from psp.data_sources.utils import _TIME, _VALUE, _VARIABLE, _X, _Y


class SatelliteDataSource(NwpDataSource):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            filter_on_step=False,
            x_dim_name="x_geostationary",
            y_dim_name="y_geostationary",
            value_name="data"
        )

        # self._data = self.prepare_data(data=self._data)
        area_definition_yaml = self._data.value.attrs["area"]

        geostationary_area_definition = pyresample.area_config.load_area_from_string(
            area_definition_yaml
        )
        geostationary_crs = geostationary_area_definition.crs

        self.lonlat_to_geostationary = pyproj.Transformer.from_crs(
            crs_from=4326,
            crs_to=geostationary_crs,
            always_xy=True,
        ).transform

    def prepare_data(self, data: xr.Dataset) -> xr.Dataset:
        # Rename the dimensions.
        rename_map: dict[str, str] = {}
        for old, new in zip(
            [
                self._x_dim_name,
                self._y_dim_name,
                self._time_dim_name,
                self._variable_dim_name,
                self._value_name,
            ],
            [_X, _Y, _TIME, _VARIABLE, _VALUE],
        ):
            if old != new:
                rename_map[old] = new

        data = data.rename(rename_map)

        # Filter data to keep only the variables in self._nwp_variables if it's not None
        if self._nwp_variables is not None:
            data = data.sel(variable=self._nwp_variables)

        return data
