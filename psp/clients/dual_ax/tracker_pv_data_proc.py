import pandas as pd
import xarray as xr
import numpy as np
import datetime as dt
import os


def main():
    os.chdir("/home/zak/dual-axis-tracking-proc/dual-axis-tracking")

    tracker_data = [
        #Enter the file paths to the tracker data here, removed for privacy
    ]

    print("Processing tracker data")
    df_t = merge_and_clean_data(tracker_data)
    print("Converting to xarray")
    ds_t = df_to_ds(df_t)
    print("Resampling and saving")
    resample_and_save(ds_t)
    print("Done")


def merge_and_clean_data(filenames):
    """
    Merge and clean multiple CSV files into a single DataFrame.

    Parameters:
    filenames (list): A list of file paths to the CSV files.

    Returns:
    df_merged (DataFrame): The merged and cleaned DataFrame.
    """
    df_list = []
    for filename in filenames:
        df = pd.read_csv(filename)
        df_list.append(df)
    
    df_merged = pd.concat(df_list)
    df_merged = df_merged.drop(columns=[" Excel Timestamp"])
    df_merged = df_merged.rename(columns={'ISO Datetime': 'ts', ' Watts': 'power'})
    df_merged['ts'] = pd.to_datetime(df_merged['ts'])

    # Keep the last as we just want the most recent values
    df_merged = df_merged.drop_duplicates(subset='ts', keep='last')
    df_merged = df_merged.set_index('ts')
    df_merged = df_merged.sort_index()

    # Filter data before September 16th, 2021 as strange values in the data
    df_merged = df_merged[df_merged.index >= '2021-09-16']
    df_merged = df_merged[df_merged.index <= '2023-12-21']

    # Get power into kw
    df_merged["power"] = df_merged["power"] / 1000

    return df_merged

def df_to_ds(df):
    """
    Convert a pandas DataFrame to an xarray Dataset.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be converted.

    Returns:
    xarray.Dataset: The converted Dataset.
    """
    ds = xr.Dataset.from_dataframe(df)
    ds = ds.set_coords('ts')

    #We only have one time series but we still give it an id.
    ds = ds.expand_dims(pv_id=[0])

    # Add hard-coded lat/lon coordinates.
    ds = ds.assign_coords(latitude=("pv_id", [51.6038]))
    ds = ds.assign_coords(longitude=("pv_id", [-1.3110]))

    # Add hard-coded tilt/orientation/capacity.
    ds = ds.assign_coords(orientation=("pv_id", [180.00]))
    ds = ds.assign_coords(tilt=("pv_id", [35.00]))

    ds = ds.assign_coords(capacity=("pv_id", [1.4]))

    return ds

def resample_and_save(ds):
    ds.to_netcdf("/mnt/leonardo/storage_b/data/ocf/solar_pv_nowcasting/clients/dual_ax/PV/dual_ax_tracker_min_kw.nc")
    print("saved minute data")

    ds_t5 = ds.resample(ts='5min', loffset=dt.timedelta(seconds=60 * 2.5)).mean()
    ds_t5.to_netcdf("/mnt/leonardo/storage_b/data/ocf/solar_pv_nowcasting/clients/dual_ax/PV/dual_ax_tracker_5min_mid_kw.nc")
    print("saved 5 minunte data")

    ds_t15 = ds.resample(ts='15min', loffset=dt.timedelta(seconds=60 * 7.5)).mean()
    ds_t15.to_netcdf("/mnt/leonardo/storage_b/data/ocf/solar_pv_nowcasting/clients/dual_ax/PV/dual_ax_tracker_15min_mid_kw.nc")
    print("saved 15 minutuely data")


if __name__ == "__main__":
    main()

