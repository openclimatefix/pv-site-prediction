import datetime as dt
import pathlib
import pickle
from typing import TypeVar

# This import registers a codec.
import ocf_blosc2  # noqa
import xarray as xr

from psp.gis import CoordinateTransformer
from psp.typings import Timestamp
from psp.utils.dates import to_pydatetime
from psp.utils.hashing import naive_hash

T = TypeVar("T", bound=xr.Dataset | xr.DataArray)

_X = "x"
_Y = "y"
_TIME = "time"
_STEP = "step"
_VARIABLE = "variable"
_VALUE = "value"


def _slice_on_lat_lon(
    data: T,
    *,
    min_lat: float | None = None,
    max_lat: float | None = None,
    min_lon: float | None = None,
    max_lon: float | None = None,
    nearest_lat: float | None = None,
    nearest_lon: float | None = None,
    transformer: CoordinateTransformer,
    x_is_ascending: bool,
    y_is_ascending: bool,
) -> T:
    # Only allow `None` values for lat/lon if they are all None (in which case we don't filter
    # by lat/lon).
    num_none = sum([x is None for x in [min_lat, max_lat, min_lon, max_lon]])
    assert num_none in [0, 4]

    if min_lat is not None:
        assert min_lat is not None
        assert min_lon is not None
        assert max_lat is not None
        assert max_lon is not None

        assert max_lat >= min_lat
        assert max_lon >= min_lon

        point1, point2 = transformer([(min_lat, min_lon), (max_lat, max_lon)])
        min_x, min_y = point1
        max_x, max_y = point2

        if not x_is_ascending:
            min_x, max_x = max_x, min_x
        if not y_is_ascending:
            min_y, max_y = max_y, min_y

        # Type ignore because this is still simpler than addin some `@overload`.
        return data.sel(x=slice(min_x, max_x), y=slice(min_y, max_y))  # type: ignore

    elif nearest_lat is not None and nearest_lon is not None:
        ((x, y),) = transformer([(nearest_lat, nearest_lon)])

        return data.sel(x=x, y=y, method="nearest")  # type: ignore

    return data


class _NwpDataSourceAtTimestamp:
    def __init__(
        self,
        data: xr.DataArray,
        transformer: CoordinateTransformer,
        x_is_ascending: bool,
        y_is_ascending: bool,
    ):
        self._data = data
        self._coordinate_transformer = transformer
        self._x_is_ascending = x_is_ascending
        self._y_is_ascending = y_is_ascending

    def get(
        self,
        timestamps: Timestamp | list[Timestamp],
        *,
        min_lat: float | None = None,
        max_lat: float | None = None,
        min_lon: float | None = None,
        max_lon: float | None = None,
        nearest_lat: float | None = None,
        nearest_lon: float | None = None,
        load: bool = False,
    ) -> xr.DataArray:
        """Slice the internal xarray."""
        if isinstance(timestamps, Timestamp):
            timestamps = [timestamps]

        for t in timestamps:
            assert t >= self.time

        # How long after `time` do we need the predictions.
        deltas = [t - self.time for t in timestamps]

        da = self._data

        da = _slice_on_lat_lon(
            da,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            nearest_lat=nearest_lat,
            nearest_lon=nearest_lon,
            transformer=self._coordinate_transformer,
            x_is_ascending=self._x_is_ascending,
            y_is_ascending=self._y_is_ascending,
        )

        # Get the nearest prediction to what we are interested in.
        da = da.sel(step=deltas, method="nearest")

        if load:
            da = da.load()
        return da

    @property
    def time(self) -> Timestamp:
        return to_pydatetime(self._data.coords[_TIME].values)


class NwpDataSource:
    """Wrapper around a zarr file containing the NWP data.

    We assume that the data is a 5D tensor, with the dimensions being (time, step, x, y, variable).

    Example:
    -------
    >>> ds = NwpDataSource('path.zarr')
    >>> ds_now = ds.at(now, min_lat=123, ..., load=True)
    >>> for prediction_time in prediction_times:
    >>>     data = ds_now.get(prediction_time)
    """

    def __init__(
        self,
        path: str,
        *,
        coord_system: int,
        x_dim_name: str = _X,
        y_dim_name: str = _Y,
        time_dim_name: str = _TIME,
        step_dim_name: str = _STEP,
        variable_dim_name: str = _VARIABLE,
        value_name: str = _VALUE,
        x_is_ascending: bool = True,
        y_is_ascending: bool = True,
        cache_dir: str | None = None,
        lag_minutes: float = 0.0,
    ):
        """
        Arguments:
        ---------
        path: Path to the .zarr data.
        coord_system: Integer representing the coordinate system for the position dimensions. 4326
            for (latitude, longitude), 27700 for OSGB, etc.
        *_dim_name: The 5 names of thedimensions in the data at `path`.
        value_name: The name of the value in the dataset at `path`.
        cache_dir: If provided, the `at_get` function will cache its result in the directory. This
            is useful when always training and testing on the same dataset, as the loading of the
            NWP is one of the main bottlenecks. Use with caution: it will create a lot of files!
        lag_minutes: Delay (in minutes) before the data is available. This is to mimic the fact that
            in production, the data is often late. We will add a "lag_minutes" of `lag_minutes`
            minutes when calling the `at` method.
        x_is_ascending: Is the `x` coordinate in ascending order. If it's in descending order, set
            this to `False`.
        y_is_ascending: Is the `y` coordinate in ascending order. If it's in descending order, set
            this to `False`.
        """
        self._path = path
        # We'll have to transform the lat/lon coordinates to the internal dataset's coordinate
        # system.
        self._coordinate_transformer = CoordinateTransformer(4326, coord_system)

        self._x_dim_name = x_dim_name
        self._y_dim_name = y_dim_name
        self._time_dim_name = time_dim_name
        self._step_dim_name = step_dim_name
        self._variable_dim_name = variable_dim_name
        self._value_name = value_name
        self._x_is_ascending = x_is_ascending
        self._y_is_ascending = y_is_ascending

        self._lag_minutes = lag_minutes

        self._open()

        self._cache_dir = pathlib.Path(cache_dir) if cache_dir else None

        if self._cache_dir:
            self._cache_dir.mkdir(exist_ok=True)

    def _open(self):
        data = xr.open_dataset(
            self._path,
            engine="zarr",
            consolidated=True,
            mode="r",
        )

        # Rename the dimensions.
        rename_map: dict[str, str] = {}
        for old, new in zip(
            [
                self._x_dim_name,
                self._y_dim_name,
                self._time_dim_name,
                self._step_dim_name,
                self._variable_dim_name,
                self._value_name,
            ],
            [_X, _Y, _TIME, _STEP, _VARIABLE, _VALUE],
        ):
            if old != new:
                rename_map[old] = new

        data = data.rename(rename_map)

        self._data = data

    def list_variables(self) -> list[str]:
        return list(self._data.coords[_VARIABLE].values)

    def at_get(
        self,
        now: Timestamp,
        *,
        nearest_lat: float,
        nearest_lon: float,
        timestamps: list[Timestamp] | Timestamp,
        load: bool = False,
    ) -> xr.DataArray:
        """Shortcut for `.at(...).get(...)`.

        This shortcut is cached if `cached_dir` was provided to the constructor.
        """
        if self._cache_dir:
            if isinstance(timestamps, Timestamp):
                timestamps = [timestamps]
            hash_data = [now, nearest_lat, nearest_lon, self._path, self._lag_minutes, *timestamps]
            hashes = tuple([naive_hash(x) for x in hash_data])
            hash_ = str(hash(hashes))
            path = self._cache_dir / hash_
            if path.exists():
                with open(path, "rb") as f:
                    data = pickle.load(f)
            else:
                data = self.at(now=now, nearest_lat=nearest_lat, nearest_lon=nearest_lon).get(
                    timestamps, load=load
                )
                with open(path, "wb") as f:
                    pickle.dump(data, f, protocol=-1)
            return data
        else:
            return self.at(now=now, nearest_lat=nearest_lat, nearest_lon=nearest_lon).get(
                timestamps, load=load
            )

    def at(
        self,
        now: Timestamp,
        *,
        min_lat: float | None = None,
        max_lat: float | None = None,
        min_lon: float | None = None,
        max_lon: float | None = None,
        nearest_lat: float | None = None,
        nearest_lon: float | None = None,
        load: bool = False,
    ) -> _NwpDataSourceAtTimestamp:
        """Slice the original data in the `time` dimension, and optionally on the x,y
        coordinates.

        Arguments:
        ---------
        min_lat: Lower bound on latitude (in degrees).
        max_lat: Upper bound on latitude.
        min_lon: Lower bound on longitude.
        max_lon: Upper bound on longitude.
        now: Time at which we are doing the query: we will use the closest "time of
            prediction" *before* this.
        load: Should we `load` the xarray in memory. It is often useful to do it at this step,
            especially when providing mix/max lat/lon coordinates: the resulting data is small
            enough and makes the subsequent queries faster.

        Return:
        ------
        A _NwpDataSourceAtTimestamp object, that can be further sliced.
        """
        ds = self._data

        ds = _slice_on_lat_lon(
            ds,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            nearest_lat=nearest_lat,
            nearest_lon=nearest_lon,
            transformer=self._coordinate_transformer,
            x_is_ascending=self._x_is_ascending,
            y_is_ascending=self._y_is_ascending,
        )

        # Forward fill so that we get the value from the past, not the future!
        ds = ds.sel(time=now - dt.timedelta(minutes=self._lag_minutes), method="ffill")
        da = ds[_VALUE]

        if load:
            da = da.load()
        return _NwpDataSourceAtTimestamp(
            da,
            self._coordinate_transformer,
            x_is_ascending=self._x_is_ascending,
            y_is_ascending=self._y_is_ascending,
        )

    def __getstate__(self):
        d = self.__dict__.copy()
        # I'm not sure of the state contained in a `Dataset` object, so I make sure we don't save
        # it.
        del d["_data"]
        return d

    def __setstate__(self, state):
        for key, value in state.items():
            setattr(self, key, value)
        self._open()
