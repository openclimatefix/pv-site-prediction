"""Simplify the pv dataset in particular for data exploration.

Given the original file, cleans it and saves it, along with sampled datasets.
"""

import argparse
import pathlib

import pandas as pd

from psp.data import C, join_effectiveness, trim_pv


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("input", type=pathlib.Path, help="input file")
    parser.add_argument("-m", "--meta", type=pathlib.Path, help="metadata.csv file")
    parser.add_argument("output", type=pathlib.Path, help="output directory")
    return parser.parse_args()


def main():
    args = parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.meta)
    df = pd.read_parquet(args.input)

    df = trim_pv(df, meta)

    df = join_effectiveness(df)

    # Rename a column.
    df = df.rename({C.POWER: "power"})

    dfs = {}

    # Make a couple of sampled datasets.
    dfs["1M"] = df.sample(1_000_000)
    dfs["10k"] = dfs["1M"].sample(10_000)

    # Keep 100 systems and make samples of those too.
    ss100 = dfs["10k"][C.ID].unique().tolist()[:100]

    dfs["100"] = df[df[C.ID].isin(ss100)]

    dfs["100_1M"] = dfs["100"].sample(1_000_000)
    dfs["100_10k"] = dfs["100_1M"].sample(10_000)

    for key, df in dfs.items():
        df.to_parquet(args.output / f"5min_{key}.parquet")


if __name__ == "__main__":
    main()
