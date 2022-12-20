import pathlib

import xarray as xr

from psp.data.uk_pv import C
from psp.ml.types import PvId, Timestamp


class PvDataSource:
    def __init__(self, filepath: pathlib.Path):
        self._path = filepath
        self.__setstate__(self.__getstate__())

    def get(
        self, pv_ids: list[PvId] | PvId, start_ts: Timestamp, end_ts: Timestamp
    ) -> xr.Dataset:
        pv_ids_list: list[PvId]
        if isinstance(pv_ids, PvId):
            pv_ids_list = [pv_ids]
        else:
            pv_ids_list = pv_ids

        return self._data.sel({C.id: pv_ids_list, C.date: slice(start_ts, end_ts)})

    def list_pv_ids(self):
        return list(self._data.coords[C.id].values)

    def min_ts(self):
        return self._data.coords[C.date].min().values[0]

    def max_ts(self):
        return self._data.coords[C.date].max().values[0]

    def __getstate__(self):
        return {"path": self._path}

    def __setstate__(self, state):
        self._path = state["path"]
        self._data = xr.open_dataset(self._path)
