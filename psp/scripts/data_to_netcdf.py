import click
import pandas as pd
import xarray as xr

from psp.data.uk_pv import C


@click.command()
@click.argument("input", type=click.Path(exists=True), required=True)
@click.argument("output", type=click.Path(exists=False), required=True)
@click.option(
    "--meta",
    "-m",
    multiple=True,
    help="Meta data to join. This argument can be used many times, the last files have precedence.",
)
def main(input, output, meta):
    """Convert a .parquet INPUT file to a .xarray file.

    The .parquet file should have been generated by simplify_data.py.
    """
    metas = meta
    metas = [pd.read_csv(m).set_index(C.id).sort_index() for m in metas]

    df = pd.read_parquet(input)

    ds = xr.Dataset.from_dataframe(
        df,
    )

    # Keep only the ss_ids that have data in all the meta files.
    # This way a given column always come only from one file.
    ss_ids_set = set(list(ds.coords["ss_id"].values))
    for m in metas:
        ss_ids_set = ss_ids_set & set(m.index.to_list())
    ss_ids = list(ss_ids_set)

    metas = [m.loc[ss_ids] for m in metas]

    # Filter the intersection of ss_ids in the dataset.
    ds = ds.sel(ss_id=ss_ids)

    columns = {"orientation", "tilt", "factor", "latitude", "longitude"}

    for meta in metas:
        ds = ds.assign_coords(
            {
                name: ([C.id], meta.loc[ss_ids, name])
                # Only consider a subset of columns.
                for name in set(meta.columns) & columns
            }
        )

    ds.to_netcdf(output)


if __name__ == "__main__":
    main()