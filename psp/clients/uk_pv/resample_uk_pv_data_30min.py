"""
This script resamples the formatted pv data to 30min intervals. The data is kept to whole
times such that the pv data for 2pm repesent the average power from 1:45pm to 2:15pm.
"""

import datetime as dt
import pandas as pd

import click
import xarray as xr


@click.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
def resample_pv_data(input_dir: str, output_dir: str) -> None:
    """
    Resample the formatted pv data.

    Args:
    ----
        input_dir (str): Directory containing the .netcdf file.
        output_dir (str): Output netcdf file.

    Returns:
    -------
        None
    """
    print("Resampling PV data...")
    ds = xr.open_dataset(input_dir)

    ds_resampled = ds.resample(timestamp="30min", base=30, label="right", closed="right").mean()

    ds_resampled.to_netcdf(output_dir)
    print("Resampled data saved to", output_dir)


if __name__ == "__main__":
    resample_pv_data()
