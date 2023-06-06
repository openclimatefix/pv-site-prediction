# pv-site-prediction

## Organisation of the repo

```
.
├── exp_reports         # Experiment reports
├── exp_results         # Default experiment result dir
├── notebooks           # Diverse notebooks
└── psp                 # Main python package
    ├── clients         # Client specific code
    ├── data_sources    # Data sources (PV, NWP, etc.)
    ├── exp_configs     # Experimentation configs
    ├── models          # Modelling code
    ├── scripts         # Scripts (entry points)
    └── tests           # Unit tests
```

## Training and evaluating a model

    poetry run python psp/scripts/train_model.py --help
    poetry run python psp/scripts/eval_model.py --help


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
