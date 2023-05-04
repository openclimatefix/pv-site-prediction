import logging
import math
import warnings
from datetime import datetime, timedelta
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from psp.data.data_sources.nwp import NwpDataSource
from psp.data.data_sources.pv import PvDataSource
from psp.models.base import PvSiteModel, PvSiteModelConfig
from psp.models.regressors.base import Regressor
from psp.pv import get_irradiance
from psp.typings import Batch, Features, Horizons, X, Y
from psp.utils.maths import safe_div

_log = logging.getLogger(__name__)


def to_midnight(ts: datetime) -> datetime:
    return ts.replace(hour=0, minute=0, second=0, microsecond=0)


def compute_history_per_horizon(
    pv_data: xr.DataArray,
    now: datetime,
    horizons: Horizons,
) -> np.ndarray:
    """Compute per-horizon averages of PV data.

    Return:
    ------
    We return a 2d matrix where rows are our horizons and columns are days. The values are the
    average PV output for that day/horizon.
    """
    # Make sure we can fit a whole number of horizons in a day. We make this assumption in a few
    # places, in particular when rolling/resampling on the PV data history in # the
    # RecentHistory model.
    assert 24 * 60 % horizons.duration == 0

    df = pv_data.to_dataframe(name="value")

    # Make sure we ignore everything before `now`.
    df = df[df.index < now]

    # Resample, matching our horizons.
    df = df.resample(timedelta(minutes=horizons.duration), origin=pd.Timestamp(now)).mean()

    df = df.reset_index()

    df["date"] = df["ts"].dt.date

    df["now"] = pd.Timestamp(now)

    df["horizon_idx"] = (
        # Get the number of seconds between the date and `now`.
        (df["ts"] - df["now"]).dt.total_seconds()
        # Remove the days.
        % (24 * 60 * 60)
        # To minutes
        / 60.0
        # How many horizon durations fit in there.
        / horizons.duration
    )

    df = pd.pivot_table(
        df,
        index="horizon_idx",
        columns="date",
        values="value",
        dropna=False,
        sort=True,
    )
    df = df.reset_index(drop=True)
    df.index.rename("horizon_idx", inplace=True)

    # Add the missing horizons. Those are the ones going after 24h.
    if len(df) < len(horizons):
        df = pd.concat([df] * math.ceil(len(horizons) / len(df)), ignore_index=True)
    df = df.iloc[: len(horizons)]

    return df.to_numpy()


def minutes_since_start_of_day(ts: datetime) -> float:
    """Time of day as minutes since midnight."""
    midnight = to_midnight(ts)
    return (ts - midnight).total_seconds() / 60.0


# To maintain backward compatibility with older serialized models, we bump this version when we make
# changes to the model. We can then adapt the `RecentHistoryModel.set_state` method to take it into
# account. It's also a good idea to add a new model fixture to the `test_load_models.py` test file
# whenever we bump this, using a simplified config file like test_config1.py (to get a small model).
_VERSION = 4


class RecentHistoryModel(PvSiteModel):
    def __init__(
        self,
        config: PvSiteModelConfig,
        pv_data_source: PvDataSource,
        nwp_data_source: NwpDataSource | None,
        regressor: Regressor,
        use_nwp: bool = True,
        nwp_variables: list[str] | None = None,
        normalize_features: bool = True,
        use_inferred_meta: bool = False,
        use_data_capacity: bool = False,
        use_capacity_as_feature: bool = True,
    ):
        self._pv_data_source: PvDataSource
        self._nwp_data_source: NwpDataSource | None
        self._regressor = regressor
        self._use_nwp = use_nwp
        self._nwp_variables = nwp_variables
        self._normalize_features = normalize_features
        self._use_inferred_meta = use_inferred_meta
        self._use_data_capacity = use_data_capacity
        self._use_capacity_as_feature = use_capacity_as_feature

        self.set_data_sources(
            pv_data_source=pv_data_source,
            nwp_data_source=nwp_data_source,
        )

        if use_nwp:
            assert self._nwp_data_source is not None

        # We bump this when we make backward-incompatible changes in the code, to support old
        # serialized models.
        self._version = _VERSION

        super().__init__(config)

    def set_data_sources(
        self, *, pv_data_source: PvDataSource, nwp_data_source: NwpDataSource | None = None
    ):
        """Set the data sources.

        This has to be called after deserializing a model using `load_model`.
        """
        self._pv_data_source = pv_data_source
        self._nwp_data_source = nwp_data_source

    def predict_from_features(self, features: Features) -> Y:
        powers = self._regressor.predict(features)
        y = Y(powers=powers)
        return y

    def get_features_with_names(self, x: X) -> Tuple[Features, dict[str, list[str]]]:
        features_with_names = self._get_features(x, with_names=True)
        assert isinstance(features_with_names, tuple)
        return features_with_names

    def get_features(self, x: X) -> Features:
        features = self._get_features(x, with_names=False)
        assert not isinstance(features, tuple)
        return features

    def _get_features(
        self, x: X, with_names: bool
    ) -> Features | Tuple[Features, dict[str, list[str]]]:
        features: Features = dict()
        data_source = self._pv_data_source.as_available_at(x.ts)

        # We'll look at stats for the previous few days.
        history_start = to_midnight(x.ts - timedelta(days=7))

        # Slice as much as we can right away.
        _data = data_source.get(
            pv_ids=x.pv_id,
            start_ts=history_start,
            end_ts=x.ts,
        )

        data = _data["power"]

        coords = _data.coords

        lat = float(coords["latitude"].values)
        lon = float(coords["longitude"].values)

        if self._use_inferred_meta:
            tilt = float(coords["tilt"].values)
            orientation = float(coords["orientation"].values)
            capacity = float(coords["factor"].values)
        else:
            # We hard-code the `tilt` and `orientation` to standard values for the UK.
            # This means that our normalization won't be perfect, but in practice the model is able
            # to cope for that.
            tilt = 35
            orientation = 180
            # We define the capacity as the 99% quantile over the last week of data.
            # In practice this seems to work well.
            try:
                # If there is a `capacity` variable in our data, we use that.
                if self._use_data_capacity:
                    # Take the first value, assuming the capacity doesn't change that rapidly.
                    capacity = _data["capacity"].values[0]
                else:
                    # Otherwise use some heuristic as capacity.
                    capacity = float(data.quantile(0.99))
            except Exception as e:
                _log.warning("Error while calculating capacity")
                _log.exception(e)
                capacity = np.nan

        # Drop every coordinate except `ts`.
        extra_var = set(data.coords).difference(["ts"])
        data = data.drop_vars(extra_var)

        # As usual we normalize the PV data wrt irradiance and our PV "factor".
        # Using `safe_div` with `np.nan` fallback to get `nan`s instead of `inf`. The `nan` are
        # ignored in `compute_history_per_horizon`.
        if self._normalize_features:
            # Get the theoretical irradiance for all the timestamps in our history.
            irr1 = get_irradiance(
                lat=lat,
                lon=lon,
                timestamps=data.coords["ts"],
                tilt=tilt,
                orientation=orientation,
            )
            norm_data = safe_div(data, irr1["poa_global"].to_numpy() * capacity, fallback=np.nan)
        else:
            norm_data = data

        history = compute_history_per_horizon(
            norm_data,
            now=x.ts,
            horizons=self.config.horizons,
        )

        # Get the middle timestamp for each of our horizons.
        horizon_timestamps = [
            x.ts + timedelta(minutes=(f0 + f1) / 2) for f0, f1 in self.config.horizons
        ]

        recent_power_minutes = 30

        # Get the irradiance for those timestamps.
        irr2 = get_irradiance(
            lat=lat,
            lon=lon,
            # We add a timestamp for the recent power, that we'll treat separately afterwards.
            timestamps=horizon_timestamps + [x.ts - timedelta(minutes=recent_power_minutes / 2)],
            tilt=tilt,
            orientation=orientation,
        )

        # TODO Should we use the other values from `get_irradiance` other than poa_global?
        poa_global: np.ndarray = irr2.loc[:, "poa_global"].to_numpy()

        poa_global_now = poa_global[-1]
        poa_global = poa_global[:-1]

        features["poa_global"] = poa_global
        features["capacity"] = capacity

        per_horizon_dict: dict[str, np.ndarray] = {
            "poa_global": poa_global,
        }

        common_dict: dict[str, float] = {}

        if self._version >= 2 and self._use_capacity_as_feature:
            common_dict["capacity"] = capacity if np.isfinite(capacity) else -1.0

        for agg in ["max", "mean", "median"]:
            # When the array is empty or all nan, numpy emits a warning. We don't care about those
            # and are happy with np.nan as a result.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
                warnings.filterwarnings("ignore", r"Mean of empty slice")
                # Use the `nan` version of the aggregators.
                aggregated = getattr(np, "nan" + agg)(history, axis=1)

            assert len(aggregated) == len(self.config.horizons)
            per_horizon_dict["h_" + agg + "_nan"] = np.isnan(aggregated) * 1.0
            per_horizon_dict["h_" + agg] = np.nan_to_num(aggregated)

        # Consider the NWP data in a small region around around our PV.
        if self._use_nwp:
            assert self._nwp_data_source is not None
            nwp_data_per_horizon = self._nwp_data_source.at_get(
                x.ts,
                nearest_lat=lat,
                nearest_lon=lon,
                timestamps=horizon_timestamps,
                load=True,
            )

            nwp_variables = self._nwp_variables or self._nwp_data_source.list_variables()
            for variable in nwp_variables:
                var_per_horizon = nwp_data_per_horizon.sel(variable=variable).values

                # We have seen `inf` values for this one.
                if variable == "vis":
                    # Using -1 because this is not a possible value for this feature.
                    var_per_horizon = np.nan_to_num(var_per_horizon, nan=-1, posinf=-1, neginf=-1.0)

                per_horizon_dict[variable] = var_per_horizon

        # Concatenate all the per-horizon features in a matrix of dimensions (horizon, features)
        per_horizon = np.stack(list(per_horizon_dict.values()), axis=-1)

        # Get the recent power.
        recent_power = float(
            data.sel(ts=slice(x.ts - timedelta(minutes=recent_power_minutes), x.ts)).mean()
        )
        recent_power_nan = np.isnan(recent_power)

        # Normalize it.
        if self._normalize_features:
            recent_power = safe_div(recent_power, poa_global_now * capacity)

        common_dict["recent_power"] = 0.0 if recent_power_nan else recent_power
        common_dict["recent_power_nan"] = recent_power_nan * 1.0

        if self._version >= 2:
            common_dict["poa_global_now_is_zero"] = poa_global_now == 0.0

        features["per_horizon"] = per_horizon
        features["common"] = np.array(list(map(float, common_dict.values())))
        if with_names:
            return (
                features,
                {
                    "per_horizon": list(per_horizon_dict.keys()),
                    "common": list(common_dict.keys()),
                },
            )
        else:
            return features

    def train(
        self, train_iter: Iterable[Batch], valid_iter: Iterable[Batch], batch_size: int
    ) -> None:
        self._regressor.train(train_iter, valid_iter, batch_size)

    def explain(self, x: X):
        """Return the internal regressor's explain."""
        features, feature_names = self.get_features_with_names(x)
        explanation = self._regressor.explain(features, feature_names)
        return explanation

    def get_state(self):
        state = self.__dict__.copy()
        # Do not save the data sources. Those should be set when loading the model using the `setup`
        # function.
        # We can't put that in __getstate__ directly because we need it when the model is pickled
        # for multiprocessing.
        del state["_pv_data_source"]
        del state["_nwp_data_source"]
        return state

    def set_state(self, state):
        if "_version" not in state:
            raise RuntimeError("You are trying to load a deprecated model")

        if state["_version"] > _VERSION:
            raise RuntimeError(
                "You are trying to load a newer model in an older version of the code"
                f" ({state['_version']} > {_VERSION})."
            )

        # Load default arguments from older versions.
        if state["_version"] < 2:
            state["_use_inferred_meta"] = True
            state["_normalize_features"] = True
        if state["_version"] < 3:
            state["_use_data_capacity"] = False
        if state["_version"] < 4:
            state["_use_capacity_as_feature"] = True
        super().set_state(state)
