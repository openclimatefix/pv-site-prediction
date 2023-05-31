# pv-site-prediction

## Organisation of the repo

[`/data`](./data)

Placeholder for data files we don't want to revision.

[`/exp_results`](./exp_results)

Placeholder for experimentation results (by default training scripts write here).

[`/notebooks`](./notebooks)

Jupyter notebooks, mostly for data exploration and experimentation analysis.

[`/psp`](./psp)

The python package defined in this repo. The bulk of the content is in here.

[`/psp/scripts`](./psp/scripts)

Different scripts for the library. For instance:

    poetry run python psp/scripts/train_model.py --help


## Prerequisites

* [poetry][poetry]


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

[poetry]: https://python-poetry.org/docs/#installation
