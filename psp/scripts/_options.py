"""Common options for a few scripts."""
import pathlib

import click

exp_root_opt = click.option(
    "-r",
    "--exp-root",
    type=click.Path(path_type=pathlib.Path),
    default="exp_results",
    help="root directory that contains experiment directories",
)

exp_name_opt = click.option(
    "-n",
    "--exp-name",
    help="name of the experiment (used as the name of the dictionary)",
)

exp_config_opt = click.option(
    "-c",
    "--exp-config-name",
    help="Experiment config file name",
    required=True,
)

num_workers_opt = click.option(
    "-w",
    "--num-workers",
    type=int,
    default=0,
    help="Number of workers for data pre-processing. Defaults to no parallelism.",
)
