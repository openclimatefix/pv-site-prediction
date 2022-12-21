"""
Data loading, cleaning, augmenting, etc. of the `uk_pv` dataset.
"""

import logging

import astral
import astral.sun
import pandas as pd

logger = logging.getLogger(__name__)


class C:
    lat = "latitude"
    lon = "longitude"
    date = "timestamp"
    power = "generation_wh"
    id = "ss_id"
    cap = "capacity"
    eff = "efficiency"


# Backward compatibility: add C.LAT, C.LON, etc.
# Should not be needed when we migrate C.LAT to C.lat
for key in dir(C):
    if not key.startswith("_"):
        setattr(C, key.upper(), getattr(C, key))


def filter_rows(pv: pd.DataFrame, mask: pd.Series, text: str | None = None):
    """Convenience method to filter a dataframe and print how much was removed."""
    n1 = len(pv)
    pv = pv[mask]
    n2 = len(pv)

    s = f"Removed {n1 - n2} ({(n1 - n2) / n1 * 100:.1f}%) rows."
    if text:
        s += f" [{text}]"
    logger.info(s)

    return pv


def trim_pv(pv: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    # Remove all the zero "power" values: there are quite a lot of those.
    pv = filter_rows(pv, pv[C.power] > 0.1, "power > 0.1")

    ss_ids = meta[C.id].unique()

    pv = filter_rows(pv, pv[C.id].isin(ss_ids), "unknown ss_id")

    return pv


def remove_nights(pv: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Remove nighttimes.

    FIXME This takes just too long to run.
    """

    lat_lon_map = {
        row[C.id]: (row[C.lat], row[C.lon]) for _, row in metadata.iterrows()
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


def get_max_power_for_time_of_day(
    df: pd.DataFrame, *, radius: int = 7, min_records: int = 0
) -> pd.DataFrame:
    """
    For each data point, find the max in a timewindow, at the same time of day.

    Arguments:
        df: index: [ss_id, timestamp], columns: [power]
        radius: How many days before and after to look at.

    Returns:
        A dataframe with the same index (but sorted!) and the max power, keeping the same column
        name.

    See the test case for an example.
    """
    df = df.reset_index(1).copy()
    df["time"] = df[C.date].dt.time
    df = df.set_index(["time", C.date], append=True, drop=False)
    # Now index is: ss_id, time, datetime

    df = df.sort_index()

    # This is where the magic happens: group by ss_id and time_of_day, then do a rolling max on the
    # days.
    df = (
        df.groupby(
            [pd.Grouper(level=0), pd.Grouper(level=1)],
        )
        .rolling(
            f"{1 + radius * 2}D",
            on=C.date,
            center=True,
            min_periods=min_records,
            closed="both",
        )
        .max()
    )

    # Reshape and sort by index.
    df = df.reset_index(level=(1, 2, 3), drop=True).sort_index()

    return df
