import abc
import datetime
import pathlib
from typing import TypeVar, overload

import numpy as np
import pandas as pd
import xarray as xr

from psp.data.uk_pv import C
from psp.ml.typings import PvId, Timestamp

# https://peps.python.org/pep-0673/
_Self = TypeVar("_Self", bound="PvDataSource")


class PvDataSource(abc.ABC):
    """Definition of the interface for loading PV data."""

    @abc.abstractmethod
    def get(
        self,
        pv_ids: list[PvId] | PvId,
        start_ts: Timestamp | None = None,
        end_ts: Timestamp | None = None,
    ) -> xr.Dataset:
        # We assume that the name of the variables in the Dataset are:
        # pv_id, power, ts, latitude, longitude, tilt, factor
        # TODO Better define `factor`.
        pass

    @abc.abstractmethod
    def list_pv_ids(self) -> list[PvId]:
        pass

    @abc.abstractmethod
    def min_ts(self) -> Timestamp:
        pass

    @abc.abstractmethod
    def max_ts(self) -> Timestamp:
        pass

    @abc.abstractmethod
    def without_future(self: _Self, ts: Timestamp, *, blackout: int = 0) -> _Self:
        """Return a copy of the data source but without the data after `ts - blackout`.

        This is a intended as a safety mechanism when we want to make sure we can't use data after
        a certain point in time. In particular, we don't want to be able to use data from the
        future when training models.

        Arguments:
        ---------
            ts: The "now" timestamp, everything after is the future. Use `None` for "don't ignore".
            blackout: A number of minutes before `ts` ("now") that we also want to ignore. Ignored
                if `ts is None`.
        """
        pass


def _min(a: Timestamp | None, b: Timestamp | None) -> Timestamp | None:
    """Util function to calculate the minimum between two timestamps that supports `None`.

    `None` values are assumed to be greater always.
    """
    if a is None:
        if b is None:
            return None
        else:
            return b
    else:
        # a is not None
        if b is None:
            return a
        else:
            return min(a, b)


# https://stackoverflow.com/questions/13703720/converting-between-datetime-timestamp-and-datetime64
def _to_py_datetime(x: np.datetime64) -> datetime.datetime:
    return pd.Timestamp(x).to_pydatetime()


class NetcdfPvDataSource(PvDataSource):
    # This constructor is used when copying.
    @overload
    def __init__(
        self, filepath: pathlib.Path, data: xr.Dataset, max_ts: Timestamp | None
    ):
        ...

    @overload
    def __init__(self, filepath: pathlib.Path | str):
        ...

    def __init__(
        self,
        filepath: pathlib.Path | str,
        data: xr.Dataset | None = None,
        max_ts: Timestamp | None = None,
    ):
        self._path = pathlib.Path(filepath)
        if data is None:
            self._open()
        else:
            self._data = data
        # See `ignore_future`.
        self._max_ts: Timestamp | None = max_ts

    def _open(self):
        self._data = xr.open_dataset(self._path).rename(
            {C.power: "power", C.date: "ts", C.id: "id"}
        )

    def get(
        self,
        pv_ids: list[PvId] | PvId,
        start_ts: Timestamp | None = None,
        end_ts: Timestamp | None = None,
    ) -> xr.Dataset:
        pv_ids_list: list[PvId]
        if isinstance(pv_ids, PvId):
            pv_ids_list = [pv_ids]
        else:
            pv_ids_list = pv_ids

        end_ts = _min(self._max_ts, end_ts)

        return self._data.sel(id=pv_ids_list, ts=slice(start_ts, end_ts))

    def list_pv_ids(self):
        return list(self._data.coords["id"].values)

    def min_ts(self):
        ts = _to_py_datetime(self._data.coords["ts"].min().values)  # type:ignore
        return _min(ts, self._max_ts)

    def max_ts(self):
        ts = _to_py_datetime(self._data.coords["ts"].max().values)  # type:ignore
        return _min(ts, self._max_ts)

    def without_future(self, ts: Timestamp, *, blackout: int = 0):
        now = ts - datetime.timedelta(minutes=blackout) - datetime.timedelta(seconds=1)
        return NetcdfPvDataSource(self._path, self._data, _min(self._max_ts, now))

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
