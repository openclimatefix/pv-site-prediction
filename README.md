# Photovoltaic (PV) Site Prediction Model

This repo contains code to train and evaluate a model to produce the forecasted energy production from solar panels (PV). It does this by providing a framework to forecast ahead, using pv data from sites, weather data (NWPs as multidimensional geospatial zarrs) and sateliite imagery (from the EUMETSAT Geostationary satellite).

## Organisation of the repo

```
.
├── exp_reports         # Experiment reports - markdown notes about experiments we have made
├── exp_results         # Default output for the {train,eval}_model.py scripts
├── notebooks           # Diverse notebooks
├── data                # Placeholder for data files
└── dashboards          # Experimental streamlit dashboards
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

## Setting up the experiement configuration

The configuration and parameters for the specific model setup is done in a python file, with the file saved under `psp/exp_configs`. In this configuration file you can:

#### Set up data sources

- Set the location for the PV data and:
  - Set the names of the different variables in this dataset.
  - Set a lag associated with using PV data in real time.
- Specify how to use the tilt, orienation and capacity metadata.
- Set the location for the NWP sources, define the coordinate system it's in, rename variables and set the variables that the model should use.
- Specificy the location for the satelitte imagery data:
  - Also indicate the size of the patch size used for satellite images.

#### Set up specific model configuration

- Set the interval between forecasts (`duration`) and the number of horizons to forecast for (`num_horizons`).
- Chose to normalise target and features.
- Select the number of training samples.
- Chose amounts to drop NWP and PV data (set data to NaNs during training) to help the model handle NaNs in production.

## PV inputs

This model forecasts the power produced by a specific solar site. If forecasting for 15 minute intervals it is best to use 15 minute data for training. To do this the you may want to resample the data. The associated timestamp which the generation represents sould be the middle of the window. More information on how the model resamples the PV data can be found in the `training.py` file.

#### Inputs to the model:

- PV features such as the generation over the last 30 minutes and what happened in the previous days. (This can be modified in the `recent_history.py` model)
- Additional PV data can be passed as a feature. Respective PV lags should be used to simulate production conditions.
- Clear sky irradiance is calculated using PVLib to give the total irradiance for the specific Plane Of Array (POA) which is used as a feature and to normalise the PV data.
- Recent PV power is added as a feature which is calcuated based on `recent_power_minutes` set in the recent_history class, where the average of data available within recent_power_minutes is used.
- `num_days_history` can also be set to help calcualte the historical mean, medium and maximum at that time over the past number of days.


## Training, Validating and Testing

Training, validation and testing can be split across different pv_ids for which the ratios can be specified in the `make_pv_splits` function in the experiment configuration. This is so that the model is trained off one of set of pv_ids and then validated and tested on an unseen set of pv_ids.

 When training, these pv_ids will be outputed. The time range is the same for teh train and validation pv_id set.

When there is only a single or small number of sites, the argument, `pv_split=None`, should be passed into the config to avoid splitting up the pv_ids.

## Inference and Backtests

When the `eval_model.py` script is run, different parameters can be passed to specify the conditions of the backtest. One option is to simulate a backtest without features generated from live PV data.

To do this a `--no-live-pv` flag can used when running the script, which will set the live PV features to NaNs. If this is used is important that the model has been trained off some NaNs from PV during training.

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
