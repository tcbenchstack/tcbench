"""
This module is taking the monolithic parquet files
generated using  ucdavis-icdm19_csv-to-parquet.py
and create random "splits" for training.

According to the logic of the paper those
splits contains 100 samples from the /pretraining
while two test splits are generated using
the original /Retraining(human-triggered) and 
/Retraining(script-triggered)
partitions
"""

import time

from typing import List

import pandas as pd
import numpy as np

import argparse
import pathlib

from tcbench.cli.richutils import rich_samples_count_report, rich_splits_report


def generate_train_splits(
    path: pathlib.Path, n_splits: int = 5, seed=12345
) -> List[pd.DataFrame]:
    """Extract n_splits of 100 samples per classes"""
    path = pathlib.Path(path)
    save_to = path.parent

    df = pd.read_parquet(path).reset_index(drop=True)

    # Quote from Sec.3.1
    # "for training set we use only 100 "triggered by script" flows per class"
    #
    # We interpret this as the training data is selected from the
    # /pretraining folder of the original dataset
    # as the other two subfolders have < 100 samples
    partition = "pretraining"
    apps = df["app"].unique()
    n_samples = 100
    rng = np.random.default_rng(seed)

    df_tmp = df[df["partition"] == partition]
    train_splits = [[] for _ in range(n_splits)]
    for app in apps:
        indexes = df_tmp[df_tmp["app"] == app].index.values
        rng.shuffle(indexes)
        indexes = indexes[: n_samples * n_splits]
        for split_indexes, l_split in zip(np.split(indexes, n_splits), train_splits):
            l_split.append(df_tmp.loc[split_indexes])

    samples_expected = len(apps) * n_samples
    for idx, l_split in enumerate(train_splits):
        split = pd.concat(l_split)
        samples_found = split.shape[0]
        assert (
            samples_found == samples_expected
        ), f"generated a split with {samples_found} samples rather than {samples_expected}"

        fname = path.parent / "imc23" / f"train_split_{idx}.parquet"
        if not fname.parent.exists():
            fname.parent.mkdir(parents=True)
        print(f"saving: {fname}")
        split.to_parquet(fname)
        train_splits[idx] = split

    split = pd.read_parquet(path.parent / "imc23" / f"train_split_0.parquet")
    ser = split["app"].value_counts()
    rich_samples_count_report(
        ser, title=f"samples count : train_split = 0 to {n_splits-1}"
    )

    return train_splits


def generate_test_splits(path: pathlib.Path) -> List[pd.DataFrame]:
    """Extract the predefined test splits from the monolithic parquet file"""
    path = pathlib.Path(path)
    # print(f"loading: {path}")
    df = pd.read_parquet(path).reset_index(drop=True)

    df_test_human = df[df["partition"] == "retraining-human-triggered"]
    df_test_script = df[df["partition"] == "retraining-script-triggered"]

    for df_tmp, name in zip((df_test_human, df_test_script), ("human", "script")):
        fname = path.parent / "imc23" / f"test_split_{name}.parquet"
        if not fname.parent.exists():
            fname.parent.mkdir(parents=True)
        print(f"\nsaving: {fname}")
        df_tmp.to_parquet(fname)

        ser = df_tmp["app"].value_counts()
        rich_samples_count_report(ser, title=f"samples count : {fname.stem}")


def main(config: dict):
    fname = (
        pathlib.Path(config["datasets"]["ucdavis-icdm19"]) / "ucdavis-icdm19.parquet"
    )
    generate_train_splits(fname)
    generate_test_splits(fname)


def cli_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        "-c",
        type=pathlib.Path,
        default="./config.yml",
        required=True,
        help="The general YAML config for dataset properties (default: %(default)s)",
    )
    return parser


if __name__ == "__main__":
    import tcbench.libtcdatasets.datasets_utils as utils

    args = cli_parser().parse_args()
    args.config = utils.load_config(args.config)

    main(args.config)
