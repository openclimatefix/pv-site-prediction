"""Takes a *simplified* dataset and rewrites it as a netCDF file."""

import argparse
import pathlib

import pandas as pd
import xarray as xr

from psp.data.uk_pv import C


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("input", type=pathlib.Path, help="input file")
    parser.add_argument("-m", "--meta", type=pathlib.Path, help="metadata.csv file")
    parser.add_argument("output", type=pathlib.Path, help="output directory")
    return parser.parse_args()


def main():
    args = parse_args()

    meta = pd.read_csv(args.meta).set_index(C.id).sort_index()

    df = pd.read_parquet(args.input)

    ds = xr.Dataset.from_dataframe(
        df,
    )

    ss_ids = list(ds.coords["ss_id"].values)

    meta = meta.loc[ss_ids]

    ds = ds.assign_coords(
        {
            name: ([C.id], meta[name])
            for name in ["latitude", "longitude", "orientation", "tilt", "kwp"]
        }
    )
    ds.to_netcdf(args.output)


if __name__ == "__main__":
    main()
