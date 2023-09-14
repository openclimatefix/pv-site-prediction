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
import pandas as pd


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

SKIP_SS_IDS = [
    str(x)
    for x in [
        8440,
        16718,
        8715,
        17073,
        9108,
        9172,
        10167,
        10205,
        10207,
        10278,
        26778,
        26819,
        10437,
        10466,
        26915,
        10547,
        26939,
        26971,
        10685,
        10689,
        2638,
        2661,
        2754,
        2777,
        2783,
        2786,
        2793,
        2812,
        2829,
        2830,
        2867,
        2883,
        2904,
        2923,
        2947,
        2976,
        2989,
        2999,
        3003,
        3086,
        3118,
        3123,
        3125,
        3264,
        3266,
        3271,
        3313,
        3334,
        3470,
        3502,
        11769,
        11828,
        11962,
        3772,
        11983,
        3866,
        3869,
        4056,
        4067,
        4116,
        4117,
        4124,
        4323,
        4420,
        20857,
        4754,
        13387,
        13415,
        5755,
        5861,
        5990,
        6026,
        6038,
        6054,
        14455,
        6383,
        6430,
        6440,
        6478,
        6488,
        6541,
        6548,
        6560,
        14786,
        6630,
        6804,
        6849,
        6868,
        6870,
        6878,
        6901,
        6971,
        7055,
        7111,
        7124,
        7132,
        7143,
        7154,
        7155,
        7156,
        7158,
        7201,
        7237,
        7268,
        7289,
        7294,
        7311,
        7329,
        7339,
        7379,
        7392,
        7479,
        7638,
        7695,
        7772,
        15967,
        7890,
        16215,
        # This one has funny night values.
        7830,
    ]
]

"""
Skip bad sites 
Relative error for Prod PV + NWP and PV: -0.056458115920149154
Relative error for Prod PV + NWP and PV + NWP + Irradiance: 0.016323565928412925
Relative error for Prod PV + NWP and PV + NWP: 0.01603146477820224
Relative error for Prod PV + NWP and PV + Irradiance: -0.05350800811237133
Relative error for PV and PV + NWP + Irradiance: 0.07278168184856208
Relative error for PV and PV + NWP: 0.0724895806983514
Relative error for PV and PV + Irradiance: 0.002950107807777826
Relative error for PV + NWP + Irradiance and PV + NWP: -0.0002921011502106825
Relative error for PV + NWP + Irradiance and PV + Irradiance: -0.06983157404078426
Relative error for PV + NWP and PV + Irradiance: -0.06953947289057356
"""

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
    #value = float(d["generation_wh"].quantile(0.99))
    #if not np.isfinite(value):
    value = float(d.coords["kwp"].values)
    return value

IRRADIANCE_DATA_PATH2 = "/run/media/jacob/data/irradiance_inference_forecast_new/"
IRRADIANCE_DATA_PATH = "/run/media/jacob/data/irradiance_inference_forecast_2021_2/"
dataset = xr.open_dataset(PV_DATA_PATH)
#print(dataset.generation_wh.max().values)
#print(dataset.generation_wh.mean().values)
#print(dataset.generation_wh.min().values)
#print(dataset.kwp.max().values)
#print(dataset.kwp)
pv_results = pd.read_csv("/home/jacob/pv-site-prediction/exp_results/pv_eval_all_sites/test_errors.csv")
combinations = ["pv_with_nwo_eval_all_sites_prod","pv_eval_all_sites","pv_with_irr_with_nwp_eval_all_sites", "pv_with_nwp_eval_all_sites", "pv_with_irr_eval_all_sites"]
combo_names = ["Prod PV + NWP","PV", "PV + NWP + Irradiance", "PV + NWP", "PV + Irradiance",]
pv_results = []
for i, combo in enumerate(combinations):
    print(combo_names[i])
    results = pd.read_csv(f"/home/jacob/pv-site-prediction/exp_results/{combo}/test_errors.csv")
    idx_to_keep = []
    ids_pv = list(set(results["pv_id"].values))
    ids_pv = [str(x) for x in ids_pv]
    # Sort ids_pv
    ids_pv.sort()
    SKIP_SS_IDS.sort()
    for idx, row in results.iterrows():
        if str(row['pv_id']) not in SKIP_SS_IDS:
            idx_to_keep.append(idx)
        else:
            print("Skipping")
    results = results.iloc[idx_to_keep]
    print(results)
    print(results.iloc[0])
    pv_results.append(results)
exit()
import glob
forecast_files = list(glob.glob(IRRADIANCE_DATA_PATH + "*.npz")) #+list(glob.glob(IRRADIANCE_DATA_PATH2 + "*.npz"))
init_times = []
for sample in tqdm.tqdm(forecast_files):
    data = np.load(sample, allow_pickle=True)
    init_time = pd.Timestamp(data["pv_metas"][0][0][0])
    init_times.append(sample.split("/")[-1].split("_")[0])
init_times = set(init_times)
forecast_files = list(glob.glob(IRRADIANCE_DATA_PATH2 + "*.npz"))
train_init_times = []
filtered_forecast_files = []
best_result_files = []
unseen_and_2021 = []
new_all = []
for sample in tqdm.tqdm(forecast_files):
    data = np.load(sample, allow_pickle=True)
    init_time = pd.Timestamp(data["pv_metas"][0][0][0])
    train_init_times.append(sample.split("/")[-1].split("_")[0])
    if sample.split("/")[-1].split("_")[0] in SKIP_SS_IDS:
        # Skip bad sites
        print("Skipping")
        continue
    new_all.append(sample)
    if init_time.year == 2021:
        unseen_and_2021.append(sample)
    if sample.split("/")[-1].split("_")[0] not in init_times:
        filtered_forecast_files.append(sample)
    else:
        # Only take ones for 2021 for the testing of sites trained on
        if init_time.year == 2021:
            best_result_files.append(sample)
forecast_files = new_all
train_init_times = set(train_init_times)
init_times = set(init_times)
print(train_init_times)
print(init_times)
print(len(train_init_times.intersection(init_times)))
print(len(train_init_times.difference(init_times)))
new_sites = train_init_times.difference(init_times)
import matplotlib.pyplot as plt
def _eval_model(forecast_files, dataset):
    """Evaluate a `model` on samples from a `dataloader` and log the error."""
    horizon_buckets = 2 * 60 # 2 hours
    errors_per_bucket = defaultdict(list)
    all_errors = []
    error_48 = []
    for sample in tqdm.tqdm(forecast_files):
        data = np.load(sample, allow_pickle=True)
        pred = data["latents"][0].clip(min=0., max=1.)
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
        #target = np.nan_to_num(target, nan=0.0)
        # Plot target and forecast
        #print(target)
        #plt.plot(target, label="target")
        #plt.plot(pred, label="pred")
        #plt.legend()init_time = pd.Timestamp(data["pv_metas"][0][0][0])
        #plt.show()
        error = abs(target - pred)
        error_48.append(error)
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
    return error_48


irradiance_inference_error = _eval_model(filtered_forecast_files, dataset)
irradiance_inference_error_all = _eval_model(forecast_files, dataset)
irradiance_inference_train = _eval_model(best_result_files, dataset)
unseen_and_2021 = _eval_model(unseen_and_2021, dataset)
# Need to take average of all the errors
irradiance_inference_error = np.nanmean(irradiance_inference_error, axis=0)
irradiance_inference_error_all = np.nanmean(irradiance_inference_error_all, axis=0)
irradiance_inference_train = np.nanmean(irradiance_inference_train, axis=0)
unseen_and_2021 = np.nanmean(unseen_and_2021, axis=0)
import matplotlib.pyplot as plt

# Plot all y and pred values for the first 48 rows
errors = [[[] for _ in range(48)],[[] for _ in range(48)],[[] for _ in range(48)],[[] for _ in range(48)],[[] for _ in range(48)]]
for i in range(0, len(pv_results[0]), 48): # 100 times
    # Plot each set of 48 points
    start_idx = i
    end_idx = i + 48
    #plt.plot(pv_results[0].iloc[start_idx:end_idx]["y"], label="target")
    for j, pv_result in enumerate(pv_results):
        #plt.plot(pv_result.iloc[start_idx:end_idx]["pred"], label=combo_names[j])
        pred_errors = pv_result.iloc[start_idx:end_idx]['error']
        for t in range(0, 48, 1):
            errors[j][t].append(np.nanmean(pred_errors.iloc[t]))
    #plt.legend()
    #plt.savefig(f"/home/jacob/Development/pv-site-prediction/exp_results/comparison_true_pv_irr/{i}.png")
    #plt.clf()

x_values = [i*15 for i in range(0, 48)]
error_comparisons = []
for i, combo in enumerate(combo_names):
    print(f"Avg error for {combo}: {np.nanmean(errors[i])}")
    avg_error = []
    for t in range(48):
        print(f"Error for period {t}-{(t+1)}: {np.nanmean(errors[i][t])}")
        avg_error.append(np.nanmean(errors[i][t]))
    plt.plot(x_values,avg_error, label=combo)
    error_comparisons.append(avg_error)
# Get the relative difference between each of the pairs errors
for i in range(len(error_comparisons)):
    for j in range(i, len(error_comparisons)):
        if i == j:
            continue
        print(f"Relative error for {combo_names[i]} and {combo_names[j]}: {np.nanmean(np.array(error_comparisons[i]) - np.array(error_comparisons[j]))}")

plt.plot(x_values, irradiance_inference_error, label="PIM PV Unseen")
plt.plot(x_values, irradiance_inference_error_all, label="PIM PV All")
plt.plot(x_values, irradiance_inference_train, label="PIM PV Train")
plt.plot(x_values, unseen_and_2021, label="PIM PV 2021")
plt.legend()
plt.xlabel("Forecast Horizon (Minutes)")
plt.ylabel("MAE")
plt.title("MAE Comparison for multiple inputs")
plt.savefig(f"/home/jacob/Development/pv-site-prediction/exp_results/avg_error_comparison_minutes_103_eval_sites_2020_2021_prod_skip_bad_sites_for_all_2_plot_all.png")

exit()
IRRADIANCE_DATA_PATH = "/run/media/jacob/data/irradiance_inference_forecast_train/"

IRRADIANCE_DATA_PATH2 = "/run/media/jacob/data/irradiance_inference_forecast_new/"
IRRADIANCE_DATA_PATH = "/run/media/jacob/data/irradiance_inference_forecast_2021_2/"
dataset = xr.open_dataset(PV_DATA_PATH)
#print(dataset.generation_wh.max().values)
#print(dataset.generation_wh.mean().values)
#print(dataset.generation_wh.min().values)
#print(dataset.kwp.max().values)
#print(dataset.kwp)
import glob
forecast_files = list(glob.glob(IRRADIANCE_DATA_PATH2 + "*.npz")) #+list(glob.glob(IRRADIANCE_DATA_PATH2 + "*.npz"))
init_times = []
for sample in tqdm.tqdm(forecast_files):
    data = np.load(sample, allow_pickle=True)
    init_time = pd.Timestamp(data["pv_metas"][0][0][0])
    init_times.append(sample.split("/")[-1].split("_")[0])

forecast_files = list(glob.glob(IRRADIANCE_DATA_PATH2 + "*.npz"))
train_init_times = []
for sample in tqdm.tqdm(forecast_files):
    data = np.load(sample, allow_pickle=True)
    init_time = pd.Timestamp(data["pv_metas"][0][0][0])
    train_init_times.append(sample.split("/")[-1].split("_")[0])

train_init_times = set(train_init_times)
init_times = set(init_times)
print(train_init_times)
print(init_times)
print(len(train_init_times.intersection(init_times)))
print(len(train_init_times.difference(init_times)))
exit()
import matplotlib.pyplot as plt
def _eval_model(forecast_files, dataset) -> None:
    """Evaluate a `model` on samples from a `dataloader` and log the error."""
    horizon_buckets = 2 * 60 # 2 hours
    errors_per_bucket = defaultdict(list)
    all_errors = []
    for sample in tqdm.tqdm(forecast_files):
        data = np.load(sample, allow_pickle=True)
        pred = data["latents"][0].clip(min=0., max=1.)
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
        #target = np.nan_to_num(target, nan=0.0)
        # Plot target and forecast
        #print(target)
        #plt.plot(target, label="target")
        #plt.plot(pred, label="pred")
        #plt.legend()
        #plt.show()
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
