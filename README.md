# pv-site-prediction

This repo contains code to train and evaluate pv-site models.

## Organisation of the repo

```
.
├── exp_reports         # Experiment reports - markdown notes about experiments we have made
├── exp_results         # Default output for the {train,eval}_model.py scripts
├── notebooks           # Diverse notebooks
├── data                # Placeholder for data files
└── psp                 # Main python package
    ├── clients         # Client specific code
    ├── data_sources    # Data sources (PV, NWP, Satellite, etc.)
    ├── exp_configs     # Experimentation configs - a config defines the different options for
    │                   # training and evaluation models. This directory contains many ready
    │                   # configs where the paths points to the data on Leonardo.
    ├── models          # The machine learning code
    ├── scripts         # Scripts (entry points)
    └── tests           # Unit tests
```

## Training and evaluating a model

    poetry run python psp/scripts/train_model.py \
        --exp-config-name test_config1 \
        -n test

    poetry run python psp/scripts/eval_model.py \
        -n test

    # This will have generated a model and test results in `exp_results/test`.

    # You can then look at the results in the `expriment_analysis.ipynb` and
    # `sample_analysis.ipynb` notebooks by setting EXP_NAMES=["test"] in the first cells.

    # Call the scripts with `--help` to see more options, in particular to run on more than one CPU.

    # The script run_exp.sh can be used to train and then evaluate a model, for example
    ./run_exp.sh exp_config_to_use name_for_exp


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
