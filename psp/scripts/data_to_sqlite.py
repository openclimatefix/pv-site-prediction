"""Creates a sqlite database from the (transformed) uk_pv data."""
import argparse
import csv
import pathlib
import sqlite3

import pyarrow.parquet as pq
import pytz
import tqdm

from psp.data.uk_pv import C


def create_tables(cur):

    cur.execute(
        """
        CREATE TABLE pv (
            pv_id INTEGER PRIMARY KEY,
            lon REAL,
            lat REAL,
            tilt REAL,
            orientation REAL,
            capacity REAL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE reading (
            reading_id INTEGER PRIMARY KEY,
            pv_id INTEGER,
            value REAL,
            ts INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE UNIQUE INDEX reading_reading_id_ts ON reading(reading_id, ts);
        """
    )

    cur.execute(
        """
        CREATE INDEX reading_pv_id ON reading(pv_id);
        """
    )

    cur.execute(
        """
        CREATE INDEX pv_lat ON pv(lat);
        """
    )

    cur.execute(
        """
        CREATE INDEX pv_lon ON pv(lon);
        """
    )


def insert_metadata(cur, rows):
    cur.executemany(
        """
        INSERT INTO pv (pv_id, lat, lon, tilt, orientation, capacity)
        VALUES(?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def insert_readings(cur, rows):
    cur.executemany(
        """
        INSERT INTO reading (pv_id, value, ts)
        VALUES(?, ?, ?)
        """,
        rows,
    )


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "-i",
        "--input",
        type=pathlib.Path,
        help="(transformed) parquet input file",
        required=True,
    )
    parser.add_argument(
        "-m", "--meta", type=pathlib.Path, help="metadata file", required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        help="output sqlite file",
        required=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_file = args.input
    metadata_file = args.meta
    sqlite_file = args.output

    con = sqlite3.connect(sqlite_file)
    cur = con.cursor()

    create_tables(cur)

    with open(metadata_file) as f:
        rows = list(csv.DictReader(f))
        rows = [
            [
                int(row[C.id]),
                float(row[C.lat]),
                float(row[C.lon]),
                float(row["tilt"]),
                float(row["orientation"]),
                float(row["kwp"]),
            ]
            for row in rows
        ]

    insert_metadata(cur, rows)

    pq_file = pq.ParquetFile(data_file)

    for batch in tqdm.tqdm(pq_file.iter_batches()):
        rows = [
            [
                row[C.id],
                row[C.power],
                row[C.date].replace(tzinfo=pytz.timezone("UTC")).timestamp(),
            ]
            for row in batch.to_pylist()
        ]
        insert_readings(cur, rows)

    con.commit()


if __name__ == "__main__":
    main()
