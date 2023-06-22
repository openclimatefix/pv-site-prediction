"""Load the raw island dataset from the POC."""

import argparse
import os

import pandas as pd
import xarray as xr


def _parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "input",
        help="folder with excel files",
    )
    parser.add_argument("output", help="output netcdf file (.nc)")

    return parser.parse_args()


# ---------- Hourly data -------------
def load_all_hourly(parent_folder_path):

    # This is a list of all of the filenames
    files = os.listdir(parent_folder_path)

    # Read all Excel files into a list of dataframes
    dataframes = []

    for filename in files:
        if filename.endswith(".xlsx"):
            file_path = os.path.join(parent_folder_path, filename)

            df = pd.read_excel(file_path, engine="openpyxl")

            # This returns a tuple
            dataframes.append(df)

        # FIXME
        # break

    all_combined_df = pd.concat(dataframes, ignore_index=True)

    # Sort the DataFrame based on the 'Datetime' column
    all_combined_df_sort = all_combined_df.sort_values(by="Date")

    return all_combined_df_sort


# 2. Convert into usable format (Transpose of hours)
def transpose_data(df):

    # Convert column names to integers
    hour_columns = [col for col in df.columns if str(col).isdigit()]

    # melt the data
    # XXX Need to retain other information, edit this
    melted = df.melt(
        id_vars=[
            "Date",
            " Total Max Capacity of Read Meters/KW",
            "Total Max Capacity",
            "Number of Read Meters",
            "Total Number of Meters",
        ],
        value_vars=hour_columns,
        var_name="Hour",
    )

    melted = melted.dropna()

    melted["Date"] = pd.to_datetime(melted["Date"])
    melted["Hour"] = pd.to_timedelta(melted["Hour"], unit="h")

    melted["Datetime"] = melted["Date"] + melted["Hour"]

    # Sort the DataFrame based on the 'Datetime' column
    melted_sorted = melted.sort_values(by="Datetime")

    melted_sorted.rename(columns={"value": "power"}, inplace=True)

    return melted_sorted


def _from_tz_and_filter(df: pd.DataFrame, time_col: str, tz: str):
    df[time_col] = (
        df[time_col].dt.tz_localize(tz, nonexistent="NaT", ambiguous="NaT").dt.tz_convert(None)
    )
    df = df[~pd.isna(df[time_col])]
    return df


def main():
    args = _parse_args()

    print("load")
    df = load_all_hourly(args.input)

    print(df.head())

    print("transpose")
    df = transpose_data(df)
    print(df.head())

    print(df.dtypes)

    df = df.rename(
        columns={
            "Datetime": "timestamp",
            "Total Max Capacity": "capacity",
        }
    )

    df["capacity"] = df["capacity"] / 1000.0
    df["power"] = df["power"] / 1000.0

    df = df[["timestamp", "power", "capacity"]]
    df = _from_tz_and_filter(df, "timestamp", "Europe/Malta")

    import datetime as dt
    d0 = dt.datetime(2021, 1, 1)
    df_1 = df[df["timestamp"] < d0]
    df_2 = df[df["timestamp"] >= d0]

    summer = (df_1["timestamp"].dt.month >= 4) & (df_1["timestamp"].dt.month <= 10)

    df_1.loc[summer, "timestamp"] = df_1.loc[summer, "timestamp"] + pd.Timedelta(hours=0.5)

    # summer_2 = (df_2["timestamp"].dt.month >= 4) & (df_2["timestamp"].dt.month <= 10)

    # df_2.loc[summer_2, "timestamp"] -= pd.Timedelta(hours=0.25)

    df = pd.concat([df_1, df_2])

    # summer_3 = (
    #     (df["timestamp"].dt.month >= 4)
    #     & (df["timestamp"].dt.month <= 10)
    #     & (df["timestamp"].dt.year >= 2020)
    # )

    # df.loc[summer_3, 'timestamp'] += pd.Timedelta(hours=0.1)

    df = df.sort_values("timestamp")

    # df["timestamp"] = df["timestamp"].dt.tz_localize("Europe/Malta", ambiguous="NaT",
    # nonexistent="NaT")

    # df = df[~pd.isna(df["timestamp"])]

    # df["timestamp"] = df["timestamp"].dt.tz_convert(None)

    # df = df[~pd.isna(df["timestamp"])]

    print(df)

    df = df.set_index("timestamp")
    ds = df.to_xarray()

    ds = ds.expand_dims(pv_id=["0"])

    # ds = ds.assign_coords(latitude=("pv_id", [35.9]))
    # ds = ds.assign_coords(longitude=("pv_id", [14.5]))

    ds = ds.assign_coords(latitude=("pv_id", [35.87420752836937]))
    ds = ds.assign_coords(longitude=("pv_id", [14.451608933898406]))

    print(ds)

    ds.to_netcdf(args.output)


if __name__ == "__main__":
    main()
