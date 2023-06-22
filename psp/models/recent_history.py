import logging
import math
import warnings
from datetime import datetime, timedelta
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import xarray as xr
from psp.data_sources.nwp import NwpDataSource
from psp.data_sources.pv import PvDataSource
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
_VERSION = 14

# To get the metadata (tilt, orientation, capacity), we need a function that takes in
# the recent PV history and returns the value.
_MetaGetter = Callable[[xr.Dataset], float]


def _default_get_tilt(*kwargs):
    return 35.0


def _default_get_orientation(*kwargs):
    return 180.0


def _default_get_capacity(d: xr.Dataset) -> float:
    return float(d["power"].quantile(0.99))


class RecentHistoryModel(PvSiteModel):
    def __init__(
        self,
        config: PvSiteModelConfig,
        *,
        pv_data_source: PvDataSource,
        nwp_data_source: NwpDataSource | None,
        regressor: Regressor,
        random_state: np.random.RandomState | None = None,
        use_nwp: bool = True,
        nwp_variables: list[str] | None = None,
        nwp_dropout: float = 0.0,
        pv_dropout: float = 0.0,
        normalize_features: bool = True,
        tilt_getter: _MetaGetter | None = None,
        orientation_getter: _MetaGetter | None = None,
        capacity_getter: _MetaGetter | None = None,
        use_capacity_as_feature: bool = True,
        num_days_history: int = 7,
        nwp_tolerance: str | None = None,
    ):
        super().__init__(config)
        # Validate some options.
        if nwp_dropout > 0.0 or pv_dropout > 0.0:
            assert random_state is not None

        if use_nwp:
            assert nwp_data_source is not None

        self._pv_data_source: PvDataSource
        self._nwp_data_source: NwpDataSource | None
        self._regressor = regressor
        self._random_state = random_state
        self._use_nwp = use_nwp
        self._nwp_variables = nwp_variables
        self._normalize_features = normalize_features

        self._pv_dropout = pv_dropout

        self._capacity_getter = capacity_getter or _default_get_capacity
        self._tilt_getter = tilt_getter or _default_get_tilt
        self._orientation_getter = orientation_getter or _default_get_orientation

        # Deprecated - keeping for backward compatibility and mypy.
        self._use_inferred_meta = None
        self._use_data_capacity = None

        self._use_capacity_as_feature = use_capacity_as_feature
        self._num_days_history = num_days_history
        self._nwp_tolerance = nwp_tolerance
        self._nwp_dropout = nwp_dropout

        self.set_data_sources(
            pv_data_source=pv_data_source,
            nwp_data_source=nwp_data_source,
        )

        # We bump this when we make backward-incompatible changes in the code, to support old
        # serialized models.
        self._version = _VERSION

        super().__init__(config)

    def set_data_sources(
        self,
        *,
        pv_data_source: PvDataSource,
        nwp_data_source: NwpDataSource | None = None,
    ):
        """Set the data sources.

        This has to be called after deserializing a model using `load_model`.
        """
        self._pv_data_source = pv_data_source
        self._nwp_data_source = nwp_data_source

    def predict_from_features(self, x: X, features: Features) -> Y:
        powers = self._regressor.predict(features)
        y = Y(powers=powers)
        return y

    def get_features_with_names(self, x: X) -> tuple[Features, dict[str, list[str]]]:
        features_with_names = self._get_features(x, with_names=True, is_training=False)
        assert isinstance(features_with_names, tuple)
        return features_with_names

    def get_features(self, x: X, is_training: bool = False) -> Features:
        features = self._get_features(x, with_names=False, is_training=is_training)
        assert not isinstance(features, tuple)
        return features

    def _get_features(
        self, x: X, with_names: bool, is_training: bool
    ) -> Features | tuple[Features, dict[str, list[str]]]:
        features: Features = dict()
        data_source = self._pv_data_source.as_available_at(x.ts)

        # We'll look at stats for the previous few days.
        history_start = to_midnight(x.ts - timedelta(days=self._num_days_history))

        # Slice as much as we can right away.
        _data = data_source.get(
            pv_ids=x.pv_id,
            start_ts=history_start,
            end_ts=x.ts,
        )

        # When there is no power value in our data (which happens mainly when we
        # explicitely make tests without power data), we make up one with NaN values.
        if "power" not in _data:
            shape = tuple(_data.dims.values())
            _data["power"] = xr.DataArray(np.empty(shape) * np.nan, dims=tuple(_data.dims))

        data = _data["power"]

        # PV data dropout.
        if (
            # Dropout makes sense only during training.
            is_training
            and self._pv_dropout > 0
            # This one is for mypy.
            and self._random_state is not None
            and self._random_state.random() < self._pv_dropout
        ):
            data *= np.nan

        coords = _data.coords

        lat = float(coords["latitude"].values)
        lon = float(coords["longitude"].values)

        # Get the metadata from the PV data.
        tilt = self._tilt_getter(_data)
        orientation = self._orientation_getter(_data)
        capacity = self._capacity_getter(_data)

        # Drop every coordinate except `ts`.
        extra_var = set(data.coords).difference(["ts"])
        data = data.drop_vars(extra_var)

        # 12: norm diffuse
        # 13: norm diffuse but add poa_global
        # 14: sins
        norm_col = "poa_diffuse" if self._version >= 12 else "poa_global"

        # As usual we normalize the PV data wrt irradiance and capacity.
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
            norm_data = safe_div(data, irr1[norm_col].to_numpy() * capacity, fallback=np.nan)
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
        poa_diffuse: np.ndarray = irr2.loc[:, "poa_diffuse"].to_numpy()

        poa_global_now = poa_global[-1]
        poa_global = poa_global[:-1]
        poa_diffuse_now = poa_diffuse[-1]
        poa_diffuse = poa_diffuse[:-1]

        features["poa_global"] = poa_diffuse if self._version >= 12 else poa_global
        features["capacity"] = capacity

        per_horizon_dict: dict[str, np.ndarray] = {
            "poa_global": poa_global,
        }

        if self._version >= 13:
            per_horizon_dict["poa_diffuse"] = poa_diffuse

        common_dict: dict[str, float] = {}

        if self._version >= 2 and self._use_capacity_as_feature:
            common_dict["capacity"] = capacity if np.isfinite(capacity) else -1.0

        if self._version >= 9:
            per_horizon_dict["hour"] = np.array(
                [
                    (x.ts + timedelta(minutes=hor)).hour
                    # FIXME add minutes
                    for hor, _ in self.config.horizons
                ]
            )

            if self._version >= 14:
                per_horizon_dict['hour'] = np.sin(per_horizon_dict['hour'] / 24 * np.pi * 2.)

        if self._version >= 11:
            common_dict["doy"] = x.ts.timetuple().tm_yday
            if self._version >= 14:
                common_dict['doy'] = np.sin(x.ts.timetuple().tm_yday / 365 * 2 * np.pi)


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

            if self._version >= 10:
                var_over_capacity = aggregated / capacity
                per_horizon_dict["h_" + agg + "_over_cap"] = np.nan_to_num(var_over_capacity)
                per_horizon_dict["h_" + agg + "_over_cap_nan"] = np.isnan(var_over_capacity) * 1.0

        # Consider the NWP data in a small region around around our PV.
        if self._use_nwp:
            assert self._nwp_data_source is not None

            # We might want to add some dropout on the NWP, *only during training*.
            if (
                is_training
                and self._nwp_dropout > 0
                # This one is for mypy.
                and self._random_state is not None
                and self._random_state.random() < self._nwp_dropout
            ):
                nwp_data_per_horizon = None
            else:
                nwp_data_per_horizon = self._nwp_data_source.get(
                    now=x.ts,
                    timestamps=horizon_timestamps,
                    nearest_lat=lat,
                    nearest_lon=lon,
                    tolerance=self._nwp_tolerance,
                )

            nwp_variables = self._nwp_variables or self._nwp_data_source.list_variables()

            for variable in nwp_variables:
                # Deal with the trivial case where the returns NWP is simply `None`. This happens if
                # there wasn't any data for the given tolerance.
                if nwp_data_per_horizon is None:
                    var_per_horizon = np.array([np.nan for _ in self.config.horizons])
                else:
                    var_per_horizon = nwp_data_per_horizon.sel(variable=variable).values

                # Deal with potential NaN values in NWP.
                var_per_horizon_is_nan = np.isnan(var_per_horizon) * 1.0
                var_per_horizon = np.nan_to_num(var_per_horizon, nan=0.0, posinf=0.0, neginf=0.0)

                per_horizon_dict[variable] = var_per_horizon
                per_horizon_dict[variable + "_isnan"] = var_per_horizon_is_nan

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

    def _v7_get_capacity(self, dataset: xr.Dataset) -> float:
        """Get capacity as it was with v7.

        Only here for backward compatibility, do not call directly.
        """
        if self._use_inferred_meta:
            return float(dataset.coords["factor"].values)
        else:
            try:
                # If there is a `capacity` variable in our data, we use that.
                if self._use_data_capacity:
                    # Take the first value, assuming the capacity doesn't change that rapidly.
                    return dataset["capacity"].values[0]
                else:
                    # Otherwise use some heuristic as capacity.
                    return float(dataset["power"].quantile(0.99))
            except Exception as e:
                _log.warning("Error while calculating capacity")
                _log.exception(e)
                return np.nan

    def _v7_get_tilt(self, dataset: xr.Dataset) -> float:
        """Get tilt as it was with v7.

        Only here for backward compatibility, do not call directly.
        """
        if self._use_inferred_meta:
            return float(dataset.coords["tilt"].values)
        else:
            return 35

    def _v7_get_orientation(self, dataset: xr.Dataset) -> float:
        """Get orientation as it was with v7.

        Only here for backward compatibility, do not call directly.
        """
        if self._use_inferred_meta:
            return float(dataset.coords["orientation"].values)
        else:
            return 180

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
        if state["_version"] < 5:
            state["_num_days_history"] = 7
        if state["_version"] < 6:
            state["_nwp_tolerance"] = None
        if state["_version"] < 7:
            state["_nwp_dropout"] = 0.0
        if state["_version"] < 8:
            state["_capacity_getter"] = self._v7_get_capacity
            state["_tilt_getter"] = self._v7_get_tilt
            state["_orientation_getter"] = self._v7_get_orientation
            state["_pv_dropout"] = 0.0

        super().set_state(state)
