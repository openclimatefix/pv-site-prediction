import numpy as np
import tqdm
import datetime as dt
import importlib
import logging
import shutil
from collections import defaultdict
from typing import TYPE_CHECKING

import click
import numpy as np
import tqdm

from psp.exp_configs.base import TrainConfigBase
from psp.metrics import mean_absolute_error
from psp.models.base import PvSiteModel
from psp.scripts._options import (
    exp_config_opt,
    exp_name_opt,
    exp_root_opt,
    log_level_opt,
    num_workers_opt,
)
from psp.serialization import save_model
from psp.typings import Sample
from psp.utils.printing import pv_list_to_short_str
import xarray as xr

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

_log = logging.getLogger(__name__)

SEED_TRAIN = 1234
SEED_VALID = 4321


def _count(x):
    """Count the number of non-nan/inf values."""
    return np.count_nonzero(np.isfinite(x))


def _err(x):
    """Calculate the error (95% confidence interval) on the mean of a list of points.

    We ignore the nan/inf values.
    """
    return 1.96 * np.nanstd(x) / np.sqrt(_count(x))

# TODO Need to open each init file, find the same init time in the PV data, then compare the timesteps
PV_DATA_PATH = "/run/media/jacob/data/5min_v3.nc"
#PV_DATA_PATH = "/mnt/storage_b/data/ocf/solar_pv_nowcasting/clients/uk_pv/5min_v3.nc"
# NWP_DATA_PATH = "gs://solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_intermediate_version_7.zarr"
NWP_DATA_PATHS = [
    (
        "/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP"
        f"/UK_Met_Office/UKV/zarr/UKV_{year}_NWP.zarr"
    )
    for year in range(2020, 2021)
]
def _get_capacity(d):
    # Use 0.99 quantile over the history window, fallback on the capacity as defined
    # in the metadata.
    value = float(d["generation_wh"].quantile(0.99))
    if not np.isfinite(value):
        value = float(d.coords["kwp"].values)
    return value
IRRADIANCE_DATA_PATH = "/run/media/jacob/data/irradiance_inference_forecast_train/"
import pandas as pd
dataset = xr.open_dataset(PV_DATA_PATH)
print(dataset.generation_wh.max().values)
print(dataset.generation_wh.mean().values)
print(dataset.generation_wh.min().values)
print(dataset.kwp.max().values)
print(dataset.kwp)
import glob
forecast_files = glob.glob(IRRADIANCE_DATA_PATH + "*.npz")
def _eval_model(forecast_files, dataset) -> None:
    """Evaluate a `model` on samples from a `dataloader` and log the error."""
    horizon_buckets = 2 * 60 # 2 hours
    errors_per_bucket = defaultdict(list)
    all_errors = []
    for sample in tqdm.tqdm(forecast_files):
        data = np.load(sample, allow_pickle=True)
        pred = data["latents"][0].clip(min=0.)
        init_time = pd.Timestamp(data["pv_metas"][0][0][0])
        sample_xr = dataset.sel(ss_id=data["location_datas"][0][0])
        # Normalize based off capacity
        capacity = _get_capacity(sample_xr)
        sample_xr = sample_xr.sel(timestamp=slice(init_time, init_time+pd.Timedelta(hours=12)))
        target = sample_xr.generation_wh.values / capacity
        if target.shape[0] != 145:
            continue
        target = target[:-1] # Remove last one to make divisble by 3, same as in training
        target = np.nanmean(target.reshape(-1, 3), axis=1) # Average over 3 timesteps
        error = abs(target - pred)
        for start, err in zip(range(0, 720, 15), error):
            bucket = start // horizon_buckets
            errors_per_bucket[bucket].append(err)
            all_errors.append(err)

    for i, errors in errors_per_bucket.items():
        bucket_start = i * horizon_buckets // 60
        bucket_end = (i + 1) * horizon_buckets // 60
        mean_err = np.nanmean(errors)
        # Error on the error!
        err_err = _err(errors)
        print(f"[{bucket_start:<2}, {bucket_end:<2}[ : {mean_err:.3f} ± {err_err:.3f}")
    mean_err = np.nanmean(all_errors)
    err_err = _err(all_errors)
    print(f"Total: {mean_err:.3f} ± {err_err:.3f}")


_eval_model(forecast_files, dataset)
