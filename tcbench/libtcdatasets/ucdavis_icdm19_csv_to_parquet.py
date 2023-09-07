#!/usr/bin/env python
# coding: utf-8
"""
This module is preprocessing the original CSV files
of the ucdavis-icdm19 paper 
(one csv file for each differen flow)
to generate a single parquet file where each 
row encodes the per-flow timeseries and 
numpy arrays

The input dataset is composed of three set 
of files in 3 subfolders: /pretraining,
/Retraining(human-triggered) and /Retraining(script-triggered)
---we call these partitions.

Within each partition, there are 5 subfolders, one for each
application.

For each application, a different file represents a different
flow (the flowid is the filename itself). Each
flow reports 4 columns corresponding to "absolute time",
"relative time to first packet", "packet size" and "direction"
of each individual packet of a flow.

The aim of this module is load all CSVs (across partitions and apps)
into a single parquet file where each row corresponds to a different
flow described by the following properties
- row_id: a unique row index
- app: one of ['google-doc', 'google-drive', 'google-music', 'google-search', 'youtube']
- flow_id: the original file name without extension
- partition: one of ['pretraining', 'retraining-human-triggered', 'retraining-script-triggered']
- num_pkts: number of packets in the flow    
- duration: duration of the flow
- bytes: number of bytes of the flow
- pkts_unixtime: np.array with the absolute time of each packet
- timetofirst: np.array with relative time of each packet with respect to the first packet
- pkts_size: np.array with each packet size
- pkts_dir: np.array with each packet direction (0 or 1)
- pkts_iat: np.array with inter packet time
"""

from rich.progress import Progress

import pandas as pd
import numpy as np

import pathlib
import itertools
import argparse
import multiprocessing

from tcbench.cli.richutils import rich_samples_count_report


def worker(fname: pathlib.Path) -> pd.DataFrame:
    """Helper function to load an individual CSV file
    into a pandas DataFrame
    """
    fname = pathlib.Path(fname)
    app = fname.parts[-2]
    partition = fname.parts[-3]

    df = pd.read_csv(
        fname,
        sep="\t",
        names=["unixtime", "timetofirst", "pkts_size", "pkts_dir"],
        dtype=dict(
            unixtime=np.float64,
            timetofirst=np.float64,
            pkts_size=np.int16,
            pkts_dir=np.int8,
        ),
    )

    df_new = pd.DataFrame(
        [
            [
                app.lower().replace(" ", "-"),  # app
                fname.stem,  # flowid
                partition.lower().replace("(", "-").replace(")", ""),  # partition
                df.shape[0],  # num_pkts
                df["timetofirst"].values[-1],  # duration
                df["pkts_size"].sum(),  # bytes
                df["unixtime"].values,  # unixtime
                df["timetofirst"].values,  # timetofirst
                df["pkts_size"].values,  # pkts_size
                df["pkts_dir"].values,  # pkts_dir
                df["timetofirst"].diff().fillna(0).values,  # pkts_iat
            ]
        ],
        columns=[
            "app",
            "flow_id",
            "partition",
            "num_pkts",
            "duration",
            "bytes",
            "unixtime",
            "timetofirst",
            "pkts_size",
            "pkts_dir",
            "pkts_iat",
        ],
    )

    return df_new


def load_files(folder: pathlib.Path, n_workers=20) -> pd.DataFrame:
    """Load all CSVs (across the 3 dataset partitions)
    using multiprocessing and concatenate them into a single DataFrame
    """
    partitions = [
        "pretraining",
        "Retraining(human-triggered)",
        "Retraining(script-triggered)",
    ]
    app_names = [
        "Google Doc",
        "Google Drive",
        "Google Music",
        "Google Search",
        "Youtube",
    ]

    # check that we have all folder
    subfolders = list(itertools.product(partitions, app_names))
    files = []
    for partition, app in subfolders:
        path = folder / partition / app
        if not path.exists():
            raise RuntimeError(f"missing {path}")
        files.extend(list(path.glob("*.txt")))

    files = sorted(files)
    print(f"found {len(files)} CSV files to load")

    from tcbench.cli import get_rich_console

    with Progress(console=get_rich_console()) as progress:
        task_id = progress.add_task("Converting CSVs...", total=len(files))
        l = []
        with multiprocessing.Pool(n_workers) as pool:
            for item in pool.imap(worker, files):
                l.append(item)
                progress.advance(task_id)
    print(f"concatenating files")

    # sorting to make sure that
    # multiprocessing does not (unintentionally)
    # breaks ordering (for replicability)
    df = pd.concat(l, axis=0).sort_values(by=["partition", "flow_id"])

    # adding a unique row
    df = df.reset_index(drop=True).reset_index().rename({"index": "row_id"}, axis=1)
    df = df.assign(app=df["app"].astype("category"))
    return df


def main(args):
    df = load_files(args.input_folder, args.num_workers)

    if not args.output_folder.exists():
        args.output_folder.mkdir(parents=True)
    fname = args.output_folder / "ucdavis-icdm19.parquet"
    print(f"saving: {fname}")
    df.to_parquet(fname)

    ser = df.groupby(["partition", "app"])["app"].count()
    rich_samples_count_report(ser, title="samples count : unfiltered")


def cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-folder",
        "-i",
        type=pathlib.Path,
        required=True,
        help="Root folder of UCDavis dataset",
    )
    parser.add_argument(
        "--output-folder",
        "-o",
        type=pathlib.Path,
        default=pathlib.Path("./datasets/ucdavis-icdm19"),
        help="Folder where to save output parquet files (default: %(default)s)",
    )
    parser.add_argument(
        "--num-workers",
        "-w",
        type=int,
        default=4,
        help="Number of workers for parallel loading (default: %(default)s)",
    )
    return parser


if __name__ == "__main__":
    parser = cli_parser()
    args = parser.parse_args()
    main(args)
