"""
Data loading, cleaning, augmenting, etc.
"""

import astral
import astral.sun
import pandas as pd


class C:
    LAT = "latitude_rounded"
    LON = "longitude_rounded"
    DATE = "timestamp"
    POWER = "generation_wh"
    ID = "ss_id"
    CAP = "capacity"
    EFF = "efficiency"


def filter_rows(pv, mask, text=None):
    n1 = len(pv)
    pv = pv[mask]
    n2 = len(pv)

    s = f"Removed {n1 - n2} ({(n1 - n2) / n1 * 100:.1f}%) rows."
    if text:
        s += f" [{text}]"
    print(s)

    return pv


def trim_pv(pv: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    # Remove all the zero "power" values: there are quite a lot of those.
    pv = filter_rows(pv, pv[C.POWER] > 0.1, "power > 0.1")

    ss_ids = meta[C.ID].unique()

    pv = filter_rows(pv, pv[C.ID].isin(ss_ids), "unknown ss_id")

    return pv


def remove_nights(pv: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Remove nighttimes.

    FIXME This takes just too long to run.
    """

    lat_lon_map = {
        row[C.ID]: (row[C.LAT], row[C.LON]) for _, row in metadata.iterrows()
    }

    pv = pv.iloc[:1_000_000].copy()

    def func(row):
        ss_id = row[C.ID]
        try:
            lat, lon = lat_lon_map[ss_id]
        except KeyError:
            return False
        date = row[C.DATE].to_pydatetime()

        obs = astral.Observer(latitude=lat, longitude=lon)
        sun_info = astral.sun.sun(obs, date)
        dawn = sun_info["dawn"]
        dusk = sun_info["dusk"]

        return dawn <= date <= dusk

    mask = pv.apply(func, axis=1)

    return pv.loc[mask]


def join_effectiveness(pv: pd.DataFrame) -> pd.DataFrame:
    """Compute some capacity for each site and normalize the power with date"""
    capacities = (
        pv[[C.ID, C.POWER]].groupby(C.ID).max().rename(columns={C.POWER: C.CAP})
    )
    pv = pv.join(capacities, on=C.ID)

    pv[C.EFF] = pv[C.POWER] / pv[C.CAP]

    return pv
