import dataclasses
import datetime
from typing import Tuple

import numpy as np

# TODO those should probably be strings in general.
PvId = int

# TODO Is this the type we want to use for dates?
# Maybe pandas' Timestamp?
Timestamp = datetime.datetime


@dataclasses.dataclass
class X:
    pv_id: PvId
    # Time at which are making the prediction.
    ts: Timestamp


FutureIntervals = list[Tuple[float, float]]


@dataclasses.dataclass
class Y:
    powers: np.ndarray
