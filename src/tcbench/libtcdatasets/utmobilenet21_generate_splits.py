#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import pathlib
import argparse

from sklearn.model_selection import train_test_split
from tcbench.cli.richutils import (
    rich_samples_count_report,
    rich_packets_report,
    rich_splits_report,
)


SEED = 12345


def filter_dataset(
    df: pd.DataFrame, min_pkts: int = 10, min_samples_per_class: int = 100
) -> pd.DataFrame:
    """Remove flows with less than 10 packets and classes with less than 100 samples"""

    # filtering out flows with less the 10 packets
    df = df[df["packets"] > min_pkts]
    filtered_samples_count = df["app"].value_counts()

    # removing classes with less than 100 samples
    valid_classes = filtered_samples_count[
        filtered_samples_count > min_samples_per_class
    ].index.tolist()
    df = df[df["app"].isin(valid_classes)]
    final_samples_count = df["app"].value_counts()
    final_samples_count = final_samples_count[final_samples_count > 0]
    final_samples_count.name = "expected_samples"

    df = df.drop("row_id", axis=1)
    df = df.reset_index(drop=True).reset_index().rename({"index": "row_id"}, axis=1)

    df = df.set_index("row_id", drop=False)
    df.index.name = None

    df = df.assign(app=df["app"].astype(str).astype("category"))
    return df


def generate_splits(
    df: pd.DataFrame, n_splits: int = 5, start_seed: int = 12345
) -> pd.DataFrame:
    splits = []
    for split_index in range(n_splits):
        y = df["app"].cat.codes

        indexes = df["row_id"]

        # generate a first 90/10 split
        seed = start_seed + split_index
        leftover_indexes, test_indexes, *_ = train_test_split(
            indexes, stratify=y, test_size=0.1, random_state=seed
        )

        # then further split 90 by another 90/10 split
        y = df.loc[leftover_indexes]["app"].cat.codes
        train_indexes, val_indexes, *_ = train_test_split(
            leftover_indexes, test_size=0.1, stratify=y, random_state=seed * 2
        )

        splits.append(
            (
                np.array(train_indexes),
                np.array(val_indexes),
                np.array(test_indexes),
                # seed,
                split_index,
            )
        )

        set1 = set(train_indexes)
        set2 = set(val_indexes)
        set3 = set(test_indexes)
        assert len(set1.intersection(set2)) == 0
        assert len(set1.intersection(set3)) == 0
        assert len(set2.intersection(set3)) == 0

    df_splits = pd.DataFrame(
        splits, columns=["train_indexes", "val_indexes", "test_indexes", "split_index"]
    )

    return df_splits


def _verify_splits(df, df_splits):
    """Double check that by pulling the samples based
    on train/val/test indexes we obtain the per-class samples count
    """

    expected_samples_count = df["app"].value_counts()
    expected_samples_count.name = "expected_samples"

    ser = df_splits.iloc[0]
    train_indexes = ser["train_indexes"]
    val_indexes = ser["val_indexes"]
    test_indexes = ser["test_indexes"]

    df_tmp = pd.DataFrame(
        (
            df.iloc[train_indexes]["app"].value_counts(),
            df.iloc[val_indexes]["app"].value_counts(),
            df.iloc[test_indexes]["app"].value_counts(),
        ),
        index=["train_samples", "val_samples", "test_samples"],
    ).T
    df_tmp = df_tmp.assign(total=df_tmp.sum(axis=1))
    df_tmp = pd.concat((df_tmp, expected_samples_count), axis=1)

    assert (df_tmp["total"] == df_tmp["expected_samples"]).all()


def main(args):
    data_folder = pathlib.Path(args.config["datasets"]["utmobilenet21"])

    output_folder = data_folder / "imc23"
    if not output_folder.exists():
        output_folder.mkdir(parents=True)

    fname = data_folder / "utmobilenet21.parquet"
    print(f"loading: {fname}")
    df = pd.read_parquet(fname)
    rich_samples_count_report(
        df["app"].value_counts(), title="samples count : unfiltered"
    )
    rich_packets_report(
        df,
        packets_colname="packets",
        title="stats : number packets per-flow (unfiltered)",
    )

    # these values are intentionally hard coded
    min_pkts = 10
    min_samples_per_class = 100
    n_splits = 5

    df_filtered = filter_dataset(df, min_pkts=min_pkts, min_samples_per_class=100)
    fname = output_folder / f"utmobilenet21_filtered_minpkts{min_pkts}.parquet"
    print(f"\nsaving: {fname}")
    df_filtered.to_parquet(fname)
    rich_samples_count_report(
        df_filtered["app"].value_counts(),
        title=f"samples count : filtered (min_pkts={min_pkts})",
    )
    rich_packets_report(
        df_filtered,
        packets_colname="packets",
        title=f"stats : number packets per-flow (min_pkts={min_pkts})",
    )

    df_splits = generate_splits(df_filtered, n_splits=5)
    fname = output_folder / f"utmobilenet21_filtered_minpkts{min_pkts}_splits.parquet"
    _verify_splits(df_filtered, df_splits)
    print(f"saving: {fname}")
    df_splits.to_parquet(fname)
    rich_splits_report(df_filtered, df_splits)


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

    main(args)
