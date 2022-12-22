import dataclasses
import datetime
from typing import Any, Mapping, Tuple

import numpy as np

# TODO those should probably be strings in general.
PvId = int

Timestamp = datetime.datetime


@dataclasses.dataclass
class X:
    """Input for a PV site models."""

    pv_id: PvId
    # Time at which are making the prediction. Typically "now".
    ts: Timestamp


# Definition of the times at which we make the predictions.
# `[[0, 15], [15, 30]]` would mean 2 predictions, one for the next 15 minutes, and one for the
# following 15 minutes.
FutureIntervals = list[Tuple[float, float]]


@dataclasses.dataclass
class Y:
    """Output for a PV site model."""

    powers: np.ndarray


Features = Mapping[str, Any]


@dataclasses.dataclass
class Sample:
    x: X
    y: Y
    features: Features


@dataclasses.dataclass
class BatchedX:
    pv_id: list[PvId]
    ts: list[Timestamp]


@dataclasses.dataclass
class BatchedY:
    powers: np.ndarray


BatchedFeatures = Mapping[str, np.ndarray]


@dataclasses.dataclass
class Batch:
    x: BatchedX
    # Note that `y.powers` here is a 2D array.
    y: BatchedY
    features: BatchedFeatures
