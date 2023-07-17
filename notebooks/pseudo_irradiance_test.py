import numpy as np
import os
from glob import glob

# Get all the train zarrs in the folder
files = glob("/run/media/jacob/data/irradiance_xarray/*_train.zarr")
files.sort()
# Get the initalization times
for f in files:
    pass


exit()

data = np.load("/run/media/jacob/data/irradiance_inference_outputs_2020/27056_2020-10-18T11:45:00.000000000.npz", allow_pickle=True)
print([k for k in data.keys()])
print(data["latents"][0].shape) # Need to average over the x and y dimensions
latents = np.mean(data['latents'][0], axis=(2, 3))
print(latents.shape)
print(data["pv_metas"][0][0])
print(data["location_datas"][0][0])
# Get the difference in the timestamps, and add a 0th difference to the beginning
timestamps = data["pv_metas"][0][0]
timestamps_diff = np.diff(timestamps)
timestamps_diff = np.insert(timestamps_diff, 0, 0).astype('timedelta64[m]')
print(timestamps_diff)
assert timestamps_diff.shape[0] == data["pv_metas"][0][0].shape[0]
exit()

# Probably want to load all ones for a given ID, then combine them into a single xarray dataset
import re

files = os.listdir("/run/media/jacob/data/irradiance_inference_outputs_2020/")

prefixes = {}

for f in files:
    m = re.search('[^_]+_', f)
    if m:
        prefix = m.group(0)
        if prefix in prefixes:
            prefixes[prefix].append(f)
        else:
            prefixes[prefix] = [f]

print(prefixes.keys())
print(len(prefixes['3324_']))
print(len(prefixes.keys()))

prefixes_train = prefixes

files = os.listdir("/run/media/jacob/data/irradiance_inference_outputs_2021/")

prefixes = {}

for f in files:
    m = re.search('[^_]+_', f)
    if m:
        prefix = m.group(0)
        if prefix in prefixes:
            prefixes[prefix].append(f)
        else:
            prefixes[prefix] = [f]

prefixes_test = prefixes

print(len(prefixes_train.keys()))
print(len(prefixes_test.keys()))
assert len(prefixes_train.keys()) == len(prefixes_test.keys())

import xarray as xr
import pandas as pd

# For each prefix, load all the files and combine them into a single xarray dataset
# Then, combine all the xarray datasets into a single xarray dataset
# Then, save the xarray dataset to disk
for prefix in prefixes_train.keys():
    print(prefix)
    files = prefixes_train[prefix]
    latents = []
    pv_metas = []
    location_datas = []
    init_times = []
    for f in files:
        data = np.load("/run/media/jacob/data/irradiance_inference_outputs_2020/" + f, allow_pickle=True)
        latents.append(np.mean(data['latents'][0], axis=(2, 3)))
        pv_metas.append(data["pv_metas"][0][0])
        location_datas.append(data["location_datas"][0][0])
        init_times.append(data["pv_metas"][0][0][0])
    latents = np.array(latents)
    pv_metas = np.array(pv_metas)
    location_datas = np.array(location_datas)
    init_times = np.array(init_times)
    # Need to get lat/lon for the site from PV Zarr
    latents = xr.DataArray(latents, dims=["idx", "feature", "forecast_horizon"], coords={"idx": np.arange(latents.shape[0]), "feature": np.arange(latents.shape[1]), "forecast_horizon": np.arange(latents.shape[2])})
    pv_metas = xr.DataArray(pv_metas, dims=["idx", "forecast_horizon"], coords={"idx": np.arange(pv_metas.shape[0]), "forecast_horizon": np.arange(pv_metas.shape[1])})
    location_datas = xr.DataArray(location_datas, dims=["idx"], coords={"idx": np.arange(location_datas.shape[0])})
    init_times = xr.DataArray(init_times, dims=["idx"], coords={"idx": np.arange(init_times.shape[0])})
    ds = xr.Dataset({"latents": latents, "pv_metas": location_datas, "forecast_time": pv_metas, "init_time": init_times})
    print(ds)
    ds.to_zarr("/run/media/jacob/data/irradiance_xarray/" + prefix + "train.zarr", mode="w", compute=True)
for prefix in prefixes_test.keys():
    print(prefix)
    files = prefixes_test[prefix]
    latents = []
    pv_metas = []
    location_datas = []
    init_times = []
    for f in files:
        data = np.load("/run/media/jacob/data/irradiance_inference_outputs_2021/" + f, allow_pickle=True)
        latents.append(np.mean(data['latents'][0], axis=(2, 3)))
        pv_metas.append(data["pv_metas"][0][0])
        location_datas.append(data["location_datas"][0][0])
        init_times.append(data["pv_metas"][0][0][0])
    latents = np.array(latents)
    pv_metas = np.array(pv_metas)
    location_datas = np.array(location_datas)
    init_times = np.array(init_times)
    # Need to get lat/lon for the site from PV Zarr
    latents = xr.DataArray(latents, dims=["idx", "feature", "forecast_horizon"], coords={"idx": np.arange(latents.shape[0]), "feature": np.arange(latents.shape[1]), "forecast_horizon": np.arange(latents.shape[2])})
    pv_metas = xr.DataArray(pv_metas, dims=["idx", "forecast_horizon"], coords={"idx": np.arange(pv_metas.shape[0]), "forecast_horizon": np.arange(pv_metas.shape[1])})
    location_datas = xr.DataArray(location_datas, dims=["idx"], coords={"idx": np.arange(location_datas.shape[0])})
    init_times = xr.DataArray(init_times, dims=["idx"], coords={"idx": np.arange(init_times.shape[0])})
    ds = xr.Dataset({"latents": latents, "pv_metas": location_datas, "forecast_time": pv_metas, "init_time": init_times})
    print(ds)
    ds.to_zarr("/run/media/jacob/data/irradiance_xarray/" + prefix + "test.zarr", mode="w", compute=True)