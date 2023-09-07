#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import pathlib
import argparse

from sklearn.model_selection import train_test_split

from tcbench.libtcdatasets.utmobilenet21_generate_splits import _verify_splits
from tcbench.cli.richutils import (
    rich_samples_count_report,
    rich_packets_report,
    rich_splits_report,
)

START_SEED = 12345

MAX_RAW_PAYLOAD_NUM_PACKETS = 20
COLUMNS_TO_KEEP_WHEN_FILTERING = [
    "row_id",
    "conn_id",
    "packet_data_l4_raw_payload",
    "flow_metadata_bf_label",
    "flow_metadata_bf_labeling_type",
    "flow_metadata_bf_l4_payload_bytes",
    "flow_metadata_bf_duration",
    "strings",
    "android_name",
    "device_name",
    "app",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "proto",
    "packets",
    "pkts_size",
    "pkts_dir",
    "timetofirst",
]


def filter_out_ack_packets(ser):
    pkts_size = ser["packet_data_l4_payload_bytes"]
    pkts_iat = ser["packet_data_iat"]
    pkts_dir = ser["packet_data_packet_dir"]

    timetofirst = pkts_iat.cumsum()
    indexes = np.where(pkts_size > 0)[0]
    return pkts_size[indexes], timetofirst[indexes], pkts_dir[indexes]


def filter_dataset(df, min_pkts: int = 10, min_samples_per_class: int = 100):
    df = df.assign(_tmp_col=df.apply(filter_out_ack_packets, axis=1))

    # adding pkts_size, timetofirst and pkts_dir (after removing ACK packets)
    df = df.assign(
        pkts_size=df["_tmp_col"].str[0],
        timetofirst=df["_tmp_col"].str[1],
        pkts_dir=df["_tmp_col"].str[2],
    ).drop("_tmp_col", axis=1)

    # compute the actual packet size after filtering out ACK packets
    df = df.assign(packets=df["pkts_size"].apply(lambda arr: arr.shape[0]))

    df = df[(df["packets"] > min_pkts) & (df["app"] != "background")]

    df = df.assign(app=df["app"].astype(str).astype("category"))

    expected_samples_count = df["app"].value_counts()
    expected_samples_count.name = "samples_expected"
    valid_classes = expected_samples_count[expected_samples_count > 100].index.tolist()

    df = df[df["app"].isin(valid_classes)]
    final_samples_count = df["app"].value_counts()
    final_samples_count = final_samples_count[final_samples_count > 0]
    final_samples_count.name = "expected_samples"

    if "row_id" in df.columns:
        df = df.drop("row_id", axis=1)

    df = df.reset_index(drop=True).reset_index().rename({"index": "row_id"}, axis=1)
    df = df.set_index("row_id", drop=False)
    df.index.name = None

    df = df.assign(app=df["app"].astype(str).astype("category"))

    assert df["row_id"].max() + 1 == df.shape[0]

    return df


def generate_global_splits(
    df, n_splits: int = 5, start_seed: int = START_SEED
) -> pd.DataFrame:
    df = df.copy()
    df = df.set_index("row_id", drop=False)

    splits = []
    for split_index in range(n_splits):
        y = df["app"].cat.codes

        indexes = df["row_id"]

        seed = START_SEED + split_index
        leftover_indexes, test_indexes, *_ = train_test_split(
            indexes, stratify=y, test_size=0.1, random_state=seed
        )

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

    df_splits = pd.DataFrame(
        splits, columns=["train_indexes", "val_indexes", "test_indexes", "split_index"]
    )
    return df_splits


def main(args):
    data_folder = pathlib.Path(args.config["datasets"]["mirage19"])

    output_folder = data_folder / "imc23"
    if not output_folder.exists():
        output_folder.mkdir(parents=True)

    fname = data_folder / "mirage19.parquet"
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

    print(f"\nfiltering min_pkts={min_pkts}...")
    df_filtered = filter_dataset(
        df, min_pkts=10, min_samples_per_class=min_samples_per_class
    )
    # removing some columns
    df_filtered = df_filtered[COLUMNS_TO_KEEP_WHEN_FILTERING]
    # clipping the list of list with raw bytes
    df_filtered = df_filtered.assign(
        packet_data_l4_raw_payload=df_filtered["packet_data_l4_raw_payload"].apply(
            lambda data: data[:MAX_RAW_PAYLOAD_NUM_PACKETS]
        )
    )
    fname = output_folder / f"mirage19_filtered_minpkts{min_pkts}.parquet"
    print(f"saving: {fname}", flush=True)
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

    df_splits = generate_global_splits(
        df_filtered, n_splits=n_splits, start_seed=START_SEED
    )
    _verify_splits(df_filtered, df_splits)

    fname = output_folder / f"mirage19_filtered_minpkts{min_pkts}_splits.parquet"
    print(f"saving: {fname}", flush=True)
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
    mdain(args)
