"""Visualization utils mostly used in notebooks."""

import datetime as dt
from typing import Literal

import altair as alt
import numpy as np
import pandas as pd
import shap
from IPython.display import display

from psp.data.data_sources.nwp import NwpDataSource
from psp.data.data_sources.pv import PvDataSource
from psp.dataset import get_y_from_x
from psp.gis import approx_add_meters_to_lat_lon
from psp.metrics import Metric, mean_absolute_error
from psp.models.base import PvSiteModel
from psp.pv import get_irradiance
from psp.typings import Horizons, Timestamp, X, Y
from psp.utils.maths import safe_div


def _make_feature_chart(
    name: str,
    feature_obj: np.ndarray,
    feature_names: list[str] | None,
    horizon_idx: int,
    num_horizons: int,
) -> alt.Chart:
    """Make a chart with all the features.

    We try to guess which chart is best depending on the shape of the data.
    """
    shape = feature_obj.shape
    ndim = len(shape)

    vline = (
        alt.Chart(pd.DataFrame(dict(horizon=[horizon_idx])))
        .mark_rule(color="red")
        .encode(x=alt.X("horizon:Q", title="horizon"))
    )

    chart: alt.Chart | None = None

    # Guess what we should plot based on the dimension.
    if ndim == 1:
        if shape[0] == num_horizons:
            chart = (
                alt.Chart(pd.DataFrame({name: feature_obj, "horizon": range(num_horizons)}))
                .mark_circle()
                .encode(x="horizon", y=name)
                .properties(height=75, width=700)
                + vline
            )
    elif ndim == 2:
        if shape[0] == num_horizons:
            if feature_names is not None:
                assert len(feature_names) == shape[1]
            data = (
                pd.DataFrame(  # type: ignore
                    feature_obj,
                    columns=None if feature_names is None else feature_names,
                )
                .stack()
                .to_frame("value")
                .reset_index(names=["horizon", "feature"])
            )
            chart = (
                alt.Chart()
                .mark_circle()
                .encode(x="horizon", y="value")
                .properties(height=50, width=700)
            )
            chart = (
                alt.layer(chart, vline, data=data)
                .facet(row="feature")
                .resolve_scale(y="independent")
            )

    return None if chart is None else chart.properties(title=name)


def time_rule(timestamp: Timestamp, text: str, align: Literal["left", "right"]) -> alt.Chart:
    """Chart of a vertical rule at a timestamp"""
    data = pd.DataFrame(dict(text=[text], timestamp=[timestamp]))
    rule = alt.Chart(data).mark_rule(color="red").encode(x="timestamp")
    text = (
        alt.Chart(data)
        .mark_text(align=align, color="red", dx=5 if align == "left" else -5)
        .encode(text="text", x="timestamp", y=alt.value(0))
    )
    return rule + text


def _make_pv_timeseries_chart(
    x: X,
    y: Y,
    pred_ts: Timestamp,
    horizons: Horizons,
    horizon_idx: int,
    pv_data_source: PvDataSource,
    padding_hours: float = 12,
    height: int = 200,
    normalize: bool = False,
) -> alt.Chart:
    """Make a timeseries chart for the PV data."""
    # Get the ground truth PV data.
    raw_data = pv_data_source.get(
        pv_ids=x.pv_id,
        start_ts=x.ts - dt.timedelta(hours=padding_hours),
        end_ts=x.ts + dt.timedelta(hours=horizons[-1][1] / 60 + padding_hours),
    )["power"]

    # Extract the meta data for the PV.
    lat = raw_data.coords["latitude"].values
    lon = raw_data.coords["longitude"].values
    factor = raw_data.coords["factor"].values
    tilt = raw_data.coords["tilt"].values
    orientation = raw_data.coords["orientation"].values

    # Reshape as a pandas dataframe.
    pv_data = raw_data.to_dataframe()[["power"]].reset_index().rename(columns={"ts": "timestamp"})

    irr_kwargs = dict(
        lat=lat,
        lon=lon,
        tilt=tilt,
        orientation=orientation,
    )

    if normalize:
        # Normalize the ground truth with respect to pvlib's irradiance.
        irr = get_irradiance(
            timestamps=pv_data["timestamp"],  # type: ignore
            **irr_kwargs,
        )["poa_global"]

        pv_data["power"] = np.clip(safe_div(pv_data["power"], irr.to_numpy() * factor), 0, 2)

    timestamps = [x.ts + dt.timedelta(minutes=h0 + (h1 - h0) / 2) for h0, h1 in horizons]

    powers = y.powers

    if normalize:
        # Normalize the predictions with respect to pvlib's irradiance.
        irr = get_irradiance(
            timestamps=timestamps,
            **irr_kwargs,
        )["poa_global"]

        powers = np.clip(safe_div(powers, irr * factor), 0, 2)

    pred_chart = (
        alt.Chart(
            pd.DataFrame(
                dict(
                    power=powers,
                    timestamp=timestamps,
                    current=[1 if i == horizon_idx else 0 for i in range(len(horizons))],
                )
            )
        )
        .mark_circle(size=14, opacity=0.8)
        .encode(
            x="timestamp",
            y="power",
            color=alt.Color("current:O", scale=alt.Scale(range=["#1f77b4", "red"]), legend=None),
            size=alt.Size("current:O", scale=alt.Scale(range=[14, 30]), legend=None),
        )
    )

    ground_truth_chart = (
        alt.Chart(pv_data)
        .mark_line(
            point=alt.OverlayMarkDef(color="black", size=10, opacity=0.8),
            color="gray",
            opacity=0.2,
        )
        .encode(x="timestamp", y="power")
        .properties(height=height, width=800)
    )

    return (
        ground_truth_chart
        + pred_chart
        + time_rule(x.ts, "now", "right")
        + time_rule(pred_ts, "prediction time", "left")
    )


def find_horizon_index(horizon: float, horizons: Horizons) -> int:
    """
    Given a `horizon` in minutes, find the index of which interval it corresponds to,
    in a `Horizons` object.
    """
    horizon_idx = 0
    for h0, h1 in horizons:
        if h0 <= horizon < h1:
            return horizon_idx
        horizon_idx += 1

    raise RuntimeError(f"Horizon {horizon} does not make sense")


def _make_nwp_heatmap(
    ts: Timestamp,
    pred_ts: Timestamp,
    lat: float,
    lon: float,
    nwp_data_source: NwpDataSource,
    radius: float = 20_000.0,
) -> dict[str, alt.Chart]:
    """Make heatmap charts for each nwp features.

    Arguments:
    ---------
    radius: How many meters to look at on each size.
    """
    [min_lat, min_lon], [max_lat, max_lon] = approx_add_meters_to_lat_lon(
        [lat, lon],
        delta_meters=np.array(
            [
                [-radius, -radius],
                [radius, radius],
            ]
        ),
    )

    # TODO: Should this be hard-coded?
    nwp_freq = dt.timedelta(hours=3)

    nwp_data = nwp_data_source.at(
        now=ts,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
    ).get(
        # Get data for three steps. 1 before, 1 at prediction time, and 1 after.
        timestamps=[pred_ts - nwp_freq, pred_ts, pred_ts + nwp_freq]
    )

    df = nwp_data.to_dataframe()[["UKV"]].reset_index()
    df["step"] = df["step"].dt.seconds / 60.0

    charts = {}

    for variable in nwp_data.coords["variable"].values:
        chart = (
            alt.Chart()
            .mark_rect()
            .encode(
                x="x:O",
                y=alt.Y("y:O", scale=alt.Scale(reverse=True)),
                color=alt.Color("UKV:Q", scale=alt.Scale(zero=False)),
            )
            .properties(height=200, width=200)
        )

        df_var = df[df["variable"] == variable]

        charts[variable] = (
            alt.layer(chart, data=df_var).facet(column="step").resolve_scale(color="independent")
        )

        mean_data = df_var[["step", "UKV"]].groupby("step").mean().reset_index()

        charts[variable + "_mean"] = (
            alt.Chart(mean_data)
            .mark_line(point=True)
            .encode(x="step", y=alt.Y("UKV", scale=alt.Scale(zero=False)))
            .properties(height=50, width=800)
        )

    return charts


def _make_explain_chart(x: X, horizon_idx: int, model: PvSiteModel):
    """Make a model explain chart for the sample."""
    shap_values, feature_names = model.explain(x)
    chart = shap.plots.force(shap_values[horizon_idx], feature_names=feature_names)
    return chart


def plot_sample(
    x: X,
    horizon_idx: int,
    model: PvSiteModel,
    pv_data_source: PvDataSource,
    nwp_data_source: NwpDataSource,
    meta: pd.DataFrame,
    metric: Metric | None = None,
    do_nwp: bool = True,
    normalize: bool = False,
):
    """Plot a sample and relevant information

    This is used in notebooks to debug models.
    """
    if metric is None:
        metric = mean_absolute_error

    y = model.predict(x)

    y_true = get_y_from_x(x=x, horizons=model.config.horizons, data_source=pv_data_source)

    if y_true is None:
        err = None
    else:
        err = metric(y_true, y)

    pv_id = x.pv_id
    ts = x.ts
    horizon = model.config.horizons[horizon_idx]

    pred_ts = ts + dt.timedelta(minutes=horizon[0] + (horizon[1] - horizon[0]) / 2)
    meta_row = meta.loc[pv_id]
    lat = float(meta_row["latitude"])
    lon = float(meta_row["longitude"])

    row_as_dict = dict(
        ts=ts,
        pred_ts=pred_ts,
        horizon=horizon,
        lat=lat,
        lon=lon,
        error=err is not None and err[horizon_idx],
        y_true=y_true.powers[horizon_idx] if y_true else None,
        y=y.powers[horizon_idx],
    )

    for key, value in row_as_dict.items():
        print(f"{key:10} {value}")

    display(_make_explain_chart(x, horizon_idx, model))

    for normalize in [False, True]:
        print(f"Normalize = {normalize}")
        display(
            _make_pv_timeseries_chart(
                x=x,
                y=y,
                pred_ts=pred_ts,
                horizons=model.config.horizons,
                horizon_idx=horizon_idx,
                pv_data_source=pv_data_source,
                padding_hours=7 * 12,
                height=150,
                normalize=normalize,
            )
        )

        display(
            _make_pv_timeseries_chart(
                x=x,
                y=y,
                pred_ts=pred_ts,
                horizons=model.config.horizons,
                horizon_idx=horizon_idx,
                pv_data_source=pv_data_source,
                padding_hours=12,
                normalize=normalize,
            )
        )

    num_horizons = len(model.config.horizons)

    if do_nwp:
        print("*** NWP ***")
        for name, chart in _make_nwp_heatmap(
            ts=ts,
            pred_ts=pred_ts,
            lat=lat,
            lon=lon,
            nwp_data_source=nwp_data_source,
        ).items():
            print(name)
            display(chart)

    print("*** FEATURES ***")
    features, feature_names = model.get_features_with_names(X(pv_id=pv_id, ts=ts))
    for key, value in features.items():
        if isinstance(value, (int, float)):
            chart = None
        else:
            chart = _make_feature_chart(
                name=key,
                feature_obj=value,
                # We assume that the `feature_names` keys will match the `features`.
                feature_names=feature_names.get(key),
                horizon_idx=horizon_idx,
                num_horizons=num_horizons,
            )
        if chart is None:
            print(key)
            print(value)
        else:
            display(chart)
