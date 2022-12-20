import argparse
import datetime
import pathlib

# import altair as alt
import numpy as np
import pandas as pd
import tqdm

from psp.data.data_sources.pv import PvDataSource
from psp.ml.metrics import MeanRelativeError, Metric, mean_absolute_error
from psp.ml.models.previous_day import PreviousDayPvSiteModel
from psp.ml.training import make_data_loader

METRICS: dict[str, Metric] = {
    "mre_cap=1": MeanRelativeError(cap=1),
    "mae": mean_absolute_error,
}


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("-p", "--pv-data", required=True, help=".netcdf PV data")
    parser.add_argument(
        "-n",
        "--exp-name",
        help="name of the experiment (used as the name of the dictionary)",
    )
    parser.add_argument("-s", "--sample", type=float, help="Sample the test samples.")
    return parser.parse_args()


def eval_model(args: argparse.Namespace):
    data_source = PvDataSource(args.pv_data)

    random_state = np.random.RandomState(1234)

    interval_size = 15
    interval_starts = [0.0, 30, 120, 24 * 60, 48 * 60]
    future_intervals = [(s, s + interval_size) for s in interval_starts]

    # TODO Specify the model in command line arguments.
    model = PreviousDayPvSiteModel(
        data_source=data_source,
        future_intervals=future_intervals,
    )

    # What do we test on?
    test_pv_ids = [
        27058,
        27059,
        27060,
        27061,
        27062,
        27063,
        27064,
        27065,
        27066,
        27067,
    ]
    min_ts = datetime.datetime(2020, 6, 1)
    max_ts = datetime.datetime(2020, 7, 1)

    # Use a torch DataLoader to create samples efficiently.
    data_loader = make_data_loader(
        data_source=data_source,
        future_intervals=future_intervals,
        pv_ids=test_pv_ids,
        min_ts=min_ts,
        max_ts=max_ts,
        random_state=random_state,
        prob_keep_sample=args.sample,
        model=model,
    )

    # Gather all errors for every samples. We'll make a DataFrame with it.
    error_rows = []

    for sample in tqdm.tqdm(data_loader):
        x = sample["x"]
        y_true = sample["y"]
        features = sample["features"]
        y_pred = model.predict(x=x, features=features)
        for metric_name, metric in METRICS.items():
            error = metric(y_true, y_pred)
            # Error is a vector
            for i, err_value in enumerate(error):
                future = future_intervals[i][0]
                error_rows.append(
                    {
                        "pv_id": x.pv_id,
                        "ts": x.ts,
                        "metric": metric_name,
                        "error": err_value,
                        "future": future,
                    }
                )

    df = pd.DataFrame.from_records(error_rows)

    exp_name = args.exp_name or datetime.datetime.now().isoformat()

    output_dir = pathlib.Path("exp_results") / exp_name
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "errors.csv")


def main():
    args = parse_args()
    eval_model(args)


if __name__ == "__main__":
    main()
