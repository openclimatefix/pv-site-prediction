import numpy as np
import os
from glob import glob
import xarray as xr

pv_data = xr.open_dataset("/run/media/jacob/data/5min_v3.nc")
print(pv_data)
data = xr.open_zarr("/run/media/jacob/data/irradiance_xarray/combined_test.zarr").sortby("pv_id")
print(data["pv_id"])
datat = xr.open_zarr("/run/media/jacob/data/irradiance_xarray/combined_train.zarr").sortby("pv_id")
print(datat["pv_id"])
print(data)
print(datat)
# Combine the two datasets on the init_time
data = xr.concat([data, datat], dim="idx").set_coords("init_time").sortby("pv_id")
data["idx"] = np.arange(len(data["idx"]))
data = data.chunk({"pv_id": 1, "idx": -1, "feature": -1})
print(data)
data.to_zarr("/run/media/jacob/data/irradiance_xarray/combined.zarr", mode="w")
exit()

ignore_ids = [
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

irradiance_ids = data["pv_id"].values
ss_ids = pv_data["ss_id"].values

new_ignore_ids = set(ss_ids).difference(set(irradiance_ids)).union(set(ignore_ids))
print(len(new_ignore_ids))
print(list(new_ignore_ids))

exit()

# Get all the train zarrs in the folder
files = glob("/run/media/jacob/data/irradiance_xarray/*_test.zarr")
files.sort()
# Get all initialization times and see which ones exist in all of the zarrs
pv_ids = []
initalization_times = []
datas = []
for f in files:
    pv_id = f.split("/")[-1].split("_")[0]
    print(pv_id)
    data = xr.open_zarr(f).isel({"idx": slice(0, 1839)})
    data = data.expand_dims({"pv_id": [int(pv_id)]})
    initalization_time = data["init_time"].values
    initalization_times.append(initalization_time)
    pv_ids.append(pv_id)
    datas.append(data)

data = xr.concat(datas, dim="pv_id").chunk({"pv_id": 1, "idx": 1})
print(data)
data.to_zarr("/run/media/jacob/data/irradiance_xarray/combined_test.zarr")
exit()
# Get the set of initialization times that are in all of the zarrs
# Iterate through and only keep times that are in all pairs of zarrs
set_of_init_times = initalization_times[0]
min_length = np.inf
print(len(initalization_times[2][0]))
for i in range(len(initalization_times)-1):
    print(len(initalization_times))
    if len(initalization_times[i][0]) < min_length:
        min_length = len(initalization_times[i])
    #print(initalization_times[i])
    #set_of_init_times = set(initalization_times[i]).intersection(set(initalization_times[i+1]))

print("Number of initialization times in all zarrs: ")
#print(len(set_of_init_times))
print("Min length of initialization times: ")
print(min_length)
print("--------------------")
#initalization_times = np.array(initalization_times)
#pv_ids = np.array(pv_ids)
print(initalization_times.shape)
print(pv_ids.shape)
print(len(initalization_times))
print(len(np.unique(initalization_times)))

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