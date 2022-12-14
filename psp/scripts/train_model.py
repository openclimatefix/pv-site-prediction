import importlib
import logging
import pickle

import click
import numpy as np

from psp.ml.dataset import split_train_test
from psp.ml.training import make_data_loader
from psp.scripts._options import (
    exp_config_opt,
    exp_name_opt,
    exp_root_opt,
    num_workers_opt,
)
from psp.utils.interupting import continue_on_interupt


@click.command()
@exp_root_opt
@exp_name_opt
@exp_config_opt
@num_workers_opt
@click.option("-b", "--batch-size", default=32)
def main(exp_root, exp_name, exp_config_name, num_workers, batch_size):

    exp_config_module = importlib.import_module(
        "." + exp_config_name, "psp.ml.exp_configs"
    )
    exp_config = exp_config_module.ExpConfig()

    random_state = np.random.RandomState(1234)

    # Load the model.
    model = exp_config.get_model()

    pv_data_source = exp_config.get_pv_data_source()

    # Dataset
    splits = split_train_test(pv_data_source)

    data_loader = make_data_loader(
        data_source=pv_data_source,
        future_intervals=model.config.future_intervals,
        split=splits.train,
        batch_size=batch_size,
        get_features=model.get_features,
        num_workers=num_workers,
        random_state=random_state,
        shuffle=True,
    )

    limit = 128

    # Ensure that way we always have the same valid set, no matter the batch size (for this we need
    # to have only whole batches).
    assert limit % batch_size == 0

    valid_data_loader = make_data_loader(
        data_source=pv_data_source,
        future_intervals=model.config.future_intervals,
        split=splits.valid,
        batch_size=batch_size,
        get_features=model.get_features,
        num_workers=num_workers,
        random_state=np.random.RandomState(4321),
        # We shuffle to get a good sample of data points.
        shuffle=True,
        limit=limit,
    )

    with continue_on_interupt(prompt=False):
        model.train(data_loader, valid_data_loader, batch_size)

    output_dir = exp_root / exp_name
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
