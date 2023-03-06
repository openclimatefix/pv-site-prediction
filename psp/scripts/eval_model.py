import datetime
import importlib
import logging

import click
import numpy as np
import pandas as pd
import torch
import tqdm

from psp.dataset import split_train_test
from psp.metrics import Metric, mean_absolute_error
from psp.scripts._options import (
    exp_config_opt,
    exp_name_opt,
    exp_root_opt,
    num_workers_opt,
)
from psp.serialization import load_model
from psp.training import make_data_loader
from psp.utils.interupting import continue_on_interupt

METRICS: dict[str, Metric] = {
    # "mre_cap=1": MeanRelativeError(cap=1),
    "mae": mean_absolute_error,
}

_log = logging.getLogger(__name__)


@click.command()
@exp_root_opt
@exp_name_opt
@exp_config_opt
@num_workers_opt
@click.option(
    "-l",
    "--limit",
    type=int,
    default=1000,
    help="Maximum number of samples to consider.",
)
def main(exp_root, exp_name, exp_config_name, num_workers, limit):
    # This fixes problems when loading files in parallel on GCP.
    # https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing
    # https://github.com/fsspec/gcsfs/issues/379
    torch.multiprocessing.set_start_method("spawn")

    exp_config_module = importlib.import_module("." + exp_config_name, "psp.exp_configs")
    exp_config = exp_config_module.ExpConfig()

    setup_config = exp_config.get_model_setup_config()
    pv_data_source = exp_config.get_pv_data_source()

    # Load the saved model.
    model_path = exp_root / exp_name / "model.pkl"
    model = load_model(model_path)

    # Model-specifi setup.
    model.setup(setup_config)

    # Setup the dataset.

    # TODO make sure the train_split from the model is consistent with the test one - we could
    # save in the model details about the training and check them here.
    splits = split_train_test(pv_data_source)
    test_split = splits.test

    _log.info(f"Testing on split: {test_split}")

    random_state = np.random.RandomState(1234)

    # Use a torch DataLoader to create samples efficiently.
    data_loader = make_data_loader(
        data_source=pv_data_source,
        horizons=model.config.horizons,
        split=test_split,
        batch_size=None,
        random_state=random_state,
        prob_keep_sample=1.0,
        get_features=model.get_features,
        num_workers=num_workers,
        shuffle=True,
        step=15,
        limit=limit,
    )

    # Gather all errors for every samples. We'll make a DataFrame with it.
    error_rows = []

    with continue_on_interupt(prompt=False):
        for i, sample in tqdm.tqdm(enumerate(data_loader), total=limit):
            x = sample.x
            y_true = sample.y
            y_pred = model.predict_from_features(features=sample.features)
            for metric_name, metric in METRICS.items():
                error = metric(y_true, y_pred)
                # Error is a vector
                for i, (err_value, y, pred) in enumerate(zip(error, y_true.powers, y_pred.powers)):
                    horizon = model.config.horizons[i][0]
                    error_rows.append(
                        {
                            "pv_id": x.pv_id,
                            "ts": x.ts,
                            "metric": metric_name,
                            "error": err_value,
                            "horizon": horizon,
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
