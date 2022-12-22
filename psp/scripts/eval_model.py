import datetime
import importlib
import pickle

import click
import numpy as np
import pandas as pd
import tqdm

from psp.ml.dataset import split_train_test
from psp.ml.metrics import Metric, mean_absolute_error
from psp.ml.models.base import PvSiteModel
from psp.ml.training import make_data_loader
from psp.scripts._options import (
    exp_config_opt,
    exp_name_opt,
    exp_root_opt,
    num_workers_opt,
)
from psp.utils.interupting import continue_on_interupt

METRICS: dict[str, Metric] = {
    # "mre_cap=1": MeanRelativeError(cap=1),
    "mae": mean_absolute_error,
}


@click.command()
@exp_root_opt
@exp_name_opt
@exp_config_opt
@num_workers_opt
@click.option("-s", "--sample", type=float, default=1, help="Sample the test samples.")
def main(exp_root, exp_name, sample, exp_config_name, num_workers):

    exp_config_module = importlib.import_module(
        "." + exp_config_name, "psp.ml.exp_configs"
    )
    exp_config = exp_config_module.ExpConfig()

    setup_config = exp_config.get_model_setup_config()
    pv_data_source = exp_config.get_pv_data_source()

    # Load the saved model.
    model_path = exp_root / exp_name / "model.pkl"
    with open(model_path, "rb") as f:
        model: PvSiteModel = pickle.load(f)

    # Model-specifi setup.
    model.setup(setup_config)

    # Setup the dataset.

    # TODO make sure the train_split from the model is consistent with the test one - we could
    # save in the model details about the training and check them here.
    splits = split_train_test(pv_data_source)
    test_split = splits.test

    random_state = np.random.RandomState(1234)

    # Use a torch DataLoader to create samples efficiently.
    data_loader = make_data_loader(
        data_source=pv_data_source,
        future_intervals=model.config.future_intervals,
        split=test_split,
        batch_size=None,
        random_state=random_state,
        prob_keep_sample=sample,
        get_features=model.get_features,
        num_workers=num_workers,
        shuffle=True,
        step=15,
    )

    # Gather all errors for every samples. We'll make a DataFrame with it.
    error_rows = []

    with continue_on_interupt(prompt=False):
        for i, batch in tqdm.tqdm(enumerate(data_loader)):
            x = batch.x
            y_true = batch.y
            y_pred = model.predict_from_features(features=batch.features)
            for metric_name, metric in METRICS.items():
                error = metric(y_true, y_pred)
                # Error is a vector
                for i, (err_value, y, pred) in enumerate(
                    zip(error, y_true.powers, y_pred.powers)
                ):
                    future = model.config.future_intervals[i][0]
                    error_rows.append(
                        {
                            "pv_id": x.pv_id,
                            "ts": x.ts,
                            "metric": metric_name,
                            "error": err_value,
                            "future": future,
                            "y": y,
                            "pred": pred,
                        }
                    )

    df = pd.DataFrame.from_records(error_rows)

    exp_name = exp_name or datetime.datetime.now().isoformat()

    output_dir = exp_root / exp_name
    print(f"Saving results to {output_dir}")
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "errors.csv")


if __name__ == "__main__":
    main()
