import importlib
import logging
import shutil
from collections import defaultdict

import click
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from psp.exp_configs.base import ExpConfigBase
from psp.metrics import mean_absolute_error
from psp.models.base import PvSiteModel
from psp.scripts._options import (
    exp_config_opt,
    exp_name_opt,
    exp_root_opt,
    log_level_opt,
    num_workers_opt,
)
from psp.serialization import save_model
from psp.training import make_data_loader
from psp.typings import Sample

_log = logging.getLogger(__name__)

SEED_TRAIN = 1234
SEED_VALID = 4321


def _count(x):
    """Count the number of non-nan/inf values."""
    return np.count_nonzero(np.isfinite(x))


def _err(x):
    """Calculate the error (95% confidence interval) on the mean of a list of points.

    We ignore the nan/inf values.
    """
    return 1.96 * np.nanstd(x) / np.sqrt(_count(x))


def _eval_model(model: PvSiteModel, dataloader: DataLoader[Sample]) -> None:
    """Evaluate a `model` on samples from a `dataloader` and log the error."""
    horizon_buckets = 8 * 60
    errors_per_bucket = defaultdict(list)
    all_errors = []
    for sample in tqdm.tqdm(dataloader):
        pred = model.predict(sample.x)
        error = mean_absolute_error(sample.y, pred)
        for (start, end), err in zip(model.config.horizons, error):
            bucket = start // horizon_buckets
            errors_per_bucket[bucket].append(err)
            all_errors.append(err)

    for i, errors in errors_per_bucket.items():
        bucket_start = i * horizon_buckets // 60
        bucket_end = (i + 1) * horizon_buckets // 60
        mean_err = np.nanmean(errors)
        # Error on the error!
        err_err = _err(errors)
        _log.info(f"[{bucket_start:<2}, {bucket_end:<2}[ : {mean_err:.3f} ± {err_err:.3f}")
    mean_err = np.nanmean(all_errors)
    err_err = _err(all_errors)
    _log.info(f"Total: {mean_err:.3f} ± {err_err:.3f}")


@click.command()
@exp_root_opt
@exp_name_opt
@exp_config_opt
@num_workers_opt
@log_level_opt
@click.option("-b", "--batch-size", default=32, show_default=True)
@click.option(
    "--num-test-samples",
    default=100,
    show_default=True,
    help="Number of samples to use to test on train and valid. Use 0 to skip completely.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Erase the output directory if it already exists",
    default=False,
    show_default=True,
)
def main(
    exp_root,
    exp_name,
    exp_config_name,
    num_workers,
    batch_size: int,
    num_test_samples: int,
    log_level: str,
    force: bool,
):
    logging.basicConfig(level=getattr(logging, log_level.upper()))

    # This fixes problems when loading files in parallel on GCP.
    # https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing
    # https://github.com/fsspec/gcsfs/issues/379
    if num_workers > 0:
        torch.multiprocessing.set_start_method("spawn")

    exp_config_module = importlib.import_module("." + exp_config_name, "psp.exp_configs")
    exp_config: ExpConfigBase = exp_config_module.ExpConfig()

    output_dir = exp_root / exp_name
    if not output_dir.exists() or force:
        output_dir.mkdir(exist_ok=True)
    else:
        raise RuntimeError(f'Output directory "{output_dir}" already exists')

    # Also copy the config into the experiment.
    shutil.copy(f"./psp/exp_configs/{exp_config_name}.py", output_dir / "config.py")

    # Load the model.
    model = exp_config.get_model()

    pv_data_source = exp_config.get_pv_data_source()

    # Dataset
    splits = exp_config.make_dataset_splits(pv_data_source)

    _log.info(f"Training on split: {splits.train}")

    data_loader_kwargs = dict(
        data_source=pv_data_source,
        horizons=model.config.horizons,
        get_features=model.get_features,
        num_workers=num_workers,
        shuffle=True,
    )

    train_data_loader = make_data_loader(
        **data_loader_kwargs,
        batch_size=batch_size,
        split=splits.train,
        random_state=np.random.RandomState(SEED_TRAIN),
    )

    limit = 128

    # Ensure that way we always have the same valid set, no matter the batch size (for this we need
    # to have only whole batches).
    assert limit % batch_size == 0

    _log.info(f"Validating on split: {splits.valid}")

    valid_data_loader = make_data_loader(
        **data_loader_kwargs,
        split=splits.valid,
        batch_size=batch_size,
        random_state=np.random.RandomState(SEED_VALID),
        # We shuffle to get a good sample of data points.
        limit=limit,
    )

    model.train(train_data_loader, valid_data_loader, batch_size)

    path = output_dir / "model.pkl"
    _log.info(f"Saving model to {path}")
    save_model(model, path)

    # Print the error on the train/valid sets.
    if num_test_samples > 0:
        _log.info("Error on the train set")
        train_data_loader2 = make_data_loader(
            **data_loader_kwargs,
            batch_size=None,
            split=splits.train,
            limit=num_test_samples,
            random_state=np.random.RandomState(SEED_TRAIN),
        )
        _eval_model(model, train_data_loader2)

        _log.info("Error on the valid set")
        valid_data_loader2 = make_data_loader(
            **data_loader_kwargs,
            batch_size=None,
            split=splits.valid,
            limit=num_test_samples,
            random_state=np.random.RandomState(SEED_VALID),
        )
        _eval_model(model, valid_data_loader2)


if __name__ == "__main__":
    main()
