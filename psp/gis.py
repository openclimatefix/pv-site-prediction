from math import radians, sqrt, cos
from typing import Sequence

# Radius of the earth in meters.
EARTH_RADIUS = 6371_000


def approx_distance(lat_lon1: Sequence[float], lat_lon2: Sequence[float]) -> float:
    """Return the approximate distance between two (lat,lon) coodinates.

    This works if they are not too far (we ignore the curvature of the earth).
    The points are expected to be in degrees and the distance is returned in meters.
    """
    lat1, lon1 = lat_lon1
    lat2, lon2 = lat_lon2

    lat1, lon1, lat2, lon2 = [radians(x) for x in [lat1, lon1, lat2, lon2]]

    lat = (lat1 + lat2) / 2

    dy = EARTH_RADIUS * (lat2 - lat1)
    dx = EARTH_RADIUS * (lon2 - lon1) * cos(lat)

    return sqrt(dx**2 + dy**2)
