# pv-site-prediction

## Organisation of the repo

[`/data`](./data)

Placeholder for data files we don't want to revision.

[`/exp_results`](./exp_results)

Placeholder for experimentation results (by default training scripts write here).

[`/notebooks`](./notebooks)

Jupyter notebooks, mostly for data exploration and experimentation analysis.

[`/psp`](./psp)

The python package defined in this repo.

[`/psp/data`](./psp/data)

Dealing with data.

[`/psp/ml`](./psp/ml)

Machine learning (datasets, models, metrics, training, etc.).

[`/psp/scripts`](./psp/scripts)

Different scripts for the library. For instance:

    poetry run python psp/scripts/train_model.py --help

[`/psp/tests`](./psp/tests)

Unit tests.

## Development

    # Installation of the dependencies.
    poetry install

    # Formatting
    make format

    # Linting
    make lint

    # Running the tests.
    make test

    # Starting the jupyter notebooks.
    make notebook
