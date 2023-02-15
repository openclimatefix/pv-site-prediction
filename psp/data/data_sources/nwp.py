import pathlib
import pickle
from typing import Tuple, TypeVar

# This import registers a codec.
import ocf_blosc2  # noqa
import pyproj
import xarray as xr

from psp.typings import Timestamp
from psp.utils.dates import to_pydatetime
from psp.utils.hashing import naive_hash

_transformer = pyproj.Transformer.from_crs(4326, 27700)


def _to_osgb(points: list[Tuple[float, float]]) -> list:
    return list(_transformer.itransform(points))


T = TypeVar("T", bound=xr.Dataset | xr.DataArray)


def _slice_on_lat_lon(
    data: T,
    *,
    min_lat: float | None = None,
    max_lat: float | None = None,
    min_lon: float | None = None,
    max_lon: float | None = None,
    nearest_lat: float | None = None,
    nearest_lon: float | None = None,
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

        point1, point2 = _to_osgb([(min_lat, min_lon), (max_lat, max_lon)])
        min_x, min_y = point1
        max_x, max_y = point2

        # Type ignore because this is still simpler than addin some `@overload`.
        # Note the reversed min_y and max_y. This is because in NWP data, the Y is reversed.
        return data.sel(x=slice(min_x, max_x), y=slice(max_y, min_y))  # type: ignore

    elif nearest_lat is not None and nearest_lon is not None:
        ((x, y),) = _to_osgb([(nearest_lat, nearest_lon)])

        return data.sel(x=x, y=y, method="nearest")  # type: ignore

    return data


class NwpDataSourceAtTimestamp:
    def __init__(self, data: xr.DataArray):
        self._data = data

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
            assert t >= self.init_time

        # How long after `init_time` do we need the predictions.
        deltas = [t - self.init_time for t in timestamps]

        da = self._data

        da = _slice_on_lat_lon(
            da,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            nearest_lat=nearest_lat,
            nearest_lon=nearest_lon,
        )

        # Get the nearest prediction to what we are interested in.
        da = da.sel(step=deltas, method="nearest")

        if load:
            da = da.load()
        return da

    @property
    def init_time(self) -> Timestamp:
        return to_pydatetime(self._data.coords["init_time"].values)


class NwpDataSource:
    """Wrapper around a zarr file containing the NWP data.

    Example:
    -------
    >>> ds = NwpDataSource('path.zarr')
    >>> ds_now = ds.at(now, min_lat=123, ..., load=True)
    >>> for prediction_time in prediction_times:
    >>>     data = ds_now.get(prediction_time)
    """

    def __init__(self, path: str, cache_dir: str | None = None):
        self._path = path
        self._open()

        self._cache_dir = pathlib.Path(cache_dir) if cache_dir else None

    def _open(self):
        self._data = xr.open_dataset(
            self._path,
            engine="zarr",
            consolidated=True,
            mode="r",
        )

    def list_variables(self) -> list[str]:
        return list(self._data.coords["variable"].values)

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
            hash_data = [now, nearest_lat, nearest_lon, *timestamps]
            hashes = tuple([naive_hash(x) for x in hash_data])
            hash_ = str(hash(hashes))
            path = self._cache_dir / hash_
            if path.exists():
                with open(path, "rb") as f:
                    data = pickle.load(f)
            else:
                data = self.at(
                    now=now, nearest_lat=nearest_lat, nearest_lon=nearest_lon
                ).get(timestamps, load=load)
                with open(path, "wb") as f:
                    pickle.dump(data, f, protocol=-1)
            return data
        else:
            return self.at(
                now=now, nearest_lat=nearest_lat, nearest_lon=nearest_lon
            ).get(timestamps, load=load)

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
    ) -> NwpDataSourceAtTimestamp:
        """Slice the original data in the `init_time` dimension, and optionally on the x,y
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
        A NwpDataSourceAtTimestamp object, that can be further sliced.
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
        )

        # Forward fill so that we get the value from the past, not the future!
        ds = ds.sel(init_time=now, method="ffill")
        da = ds["UKV"]

        if load:
            da = da.load()
        return NwpDataSourceAtTimestamp(da)

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
