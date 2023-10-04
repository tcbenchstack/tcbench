#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import rich.progress as richprogress

import json
import re
import string
import pathlib
import argparse
import shutil
import tempfile
import functools

from typing import List, Dict
from multiprocessing import Pool
from datetime import datetime

VALIDCHARS = set(
    string.ascii_letters + string.whitespace + string.punctuation + string.digits
)

from tcbench import cli

console = cli.console


def scan_for_strings(raw_packets: List, n_packets: int, min_len: int = 5) -> List[str]:
    """Extract ASCII strings by processing packets payload

    Arguments:
        raw_packets: a list of list of raw packets bytes. The outer list
            represent a packet, the inner list the individual bytes (as integer values)
            of the payload
        n_packets: maximum number of packets to process
        min_len: minimum lenght of the strings to return

    Return:
        An array containing the identified string
    """
    strings = []
    for pkt in raw_packets[:n_packets]:
        if not pkt:
            continue
        text = "".join(char if char in VALIDCHARS else "#" for char in map(chr, pkt))
        for string in re.findall(r"[^#]+", text):
            if "http:" in string or (
                len(string) >= min_len and ("." in string or " " in string)
            ):
                strings.append(string)
    return strings


def flatten_dict(data: Dict, parent_name: str = None) -> List:
    """Helper function to flatten a nested dictionary.
    For example, the structure {"a":{"b":{"c":1, "d":2}}}
    is transformed into [("a_b_c":1), ("a_b_d":2)]

    Arguments:
        data: the dictionary to explore
        parent_name: the key associated the data
            currently processed

    Return:
        A list of (str, value) pairs where
        the str is the flattened key corresponding
        the the chaining of the nested key dictionary
    """
    if not isinstance(data, dict):
        return [(parent_name, data)]

    new_items = []
    for key, value in data.items():
        new_key = key if not parent_name else f"{parent_name}_{key}"
        new_items.extend(flatten_dict(value, new_key))
    return new_items


def convert_to_dataframe(data: Dict) -> pd.DataFrame:
    """Convert a nested dictionary into a pandas DataFrame

    Argument:
        data: a (possibly) nested dictionary of key-value pairs

    Return:
        A pandas DataFrame collecting the flattened keys
        of the nested dictionary and the related value
    """
    new_data = dict()
    for net_tuple, value in data.items():
        new_data[net_tuple] = dict(flatten_dict(value))

    df = pd.DataFrame(new_data).T
    df.columns = [col.lower() for col in df.columns]
    return df


# def worker(fname: str, progress: bool = True, save_to: str = None) -> pd.DataFrame:
def worker(fname: str, save_to: str = None) -> pd.DataFrame:
    """A helper function to transform a JSON MIRAGE input
    file into a pandas DataFrame

    Arguments:
        fname: the JSON file to process
        save_to: an optional file name where to store the
            transformed data as parquet

    Return:
        A pandas DataFrame with the loaded JSON data
    """
    with open(str(fname), "r") as fin:
        data = json.load(fin)

    for net_tuple in data:
        payload_bytes = data[net_tuple]["packet_data"]["L4_raw_payload"]
        data[net_tuple]["strings"] = scan_for_strings(payload_bytes, 10, 5)

    df = convert_to_dataframe(data)

    # extract name from filename
    android_name = "None"
    if "None" not in fname.name:
        _1, android_name, *_ = fname.name.split("_")
    df = df.assign(
        android_name=android_name,
        device_name=fname.parent.name,
    )

    #    if progress:
    #        print(f".", end="", flush=True)

    if save_to:
        out_fname = save_to / fname.parent.name / fname.name
        if not out_fname.parent.exists():
            out_fname.parent.mkdir(parents=True)
        out_fname = out_fname.with_suffix(".parquet")

    if out_fname.exists():
        raise RuntimeError(f"file {out_fname} already exists!")

    df.to_parquet(out_fname)


INVALID_IPS = {"127.0.0.1", "255.255.255.255"}


def postprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Process a the data loaded from MIRAGE JSON to
    (1) identify a background class;
    (2) remove invalid IPs (e.g., 127.0.0.1, 255.255.255.255);
    (3) add a unique "row_id" row;
    (4) add an "app" column with the label represented as category

    Argument:
        df: the pandas DataFrame to process

    Return:
        The modified version of the input data
    """
    # create a background class
    df = df.assign(
        app=np.where(
            df["android_name"] == df["flow_metadata_bf_label"],
            df["android_name"],
            "background",
        )
    )

    # split connection index to recover network tuple info
    df = df.reset_index().rename({"index": "conn_id"}, axis=1)
    df = df.assign(_tmp_col=df["conn_id"].str.split(","))
    df = df.assign(
        src_ip=df["_tmp_col"].str[0],
        src_port=df["_tmp_col"].str[1],
        dst_ip=df["_tmp_col"].str[2],
        dst_port=df["_tmp_col"].str[3],
        proto=df["_tmp_col"].str[4],
    ).drop("_tmp_col", axis=1)

    # drop invalid IPs
    # df = df[(~df["src_ip"].isin(INVALID_IPS)) & (~df["dst_ip"].isin(INVALID_IPS))]

    # add a unique row_id
    df = df.reset_index().rename({"index": "row_id"}, axis=1)

    # enforce app to be categorical
    df = df.assign(
        app=df["app"].astype("category"),
        packets=df["packet_data_l4_payload_bytes"].apply(len),
    )

    return df


def main(
    input_folder: str, save_as: pathlib.Path = None, workers: int = 30
) -> pd.DataFrame:
    """The main processing loop

    Argument:
        input_folder: the folder where the MIRAGE JSON files are contained
        save_as: the output filename where to store the parquet
            after loading the data
        workers: number of parallel workers to use for processing
    """
    input_folder = pathlib.Path(input_folder)
    if (input_folder / "MIRAGE-2019_traffic_dataset_downloadable").exists():
        input_folder /= "MIRAGE-2019_traffic_dataset_downloadable"

    # creating a temporary folder
    with tempfile.TemporaryDirectory() as tmp_folder:
        tmp_folder = pathlib.Path(tmp_folder)

        files = list(input_folder.glob("*/*.json"))
        if len(files) == 0:
            raise RuntimeError(
                f"Did not find any .json file for input folder {input_folder} ! Make sure the input folder support a */*.json glob search"
            )

        print(f"found {len(files)} JSON files to load")

        func_worker = functools.partial(worker, save_to=tmp_folder)
        # params = []
        # for path in files:
        #    params.append(
        #        (
        #            path,
        #            True,
        #            tmp_folder,
        #        )
        #    )

        with richprogress.Progress(
            richprogress.TextColumn("[progress.description]{task.description}"),
            richprogress.BarColumn(),
            richprogress.MofNCompleteColumn(),
            richprogress.TimeElapsedColumn(),
            console=console,
        ) as progressbar:
            task_id = progressbar.add_task("Converting JSONs...", total=len(files))
            with Pool(workers) as pool:
                # note: order does not matter because the worker
                #   save output to file
                for _ in pool.imap_unordered(func_worker, files):
                    progressbar.advance(task_id)

        print("merging files...")
        l = list(map(pd.read_parquet, tmp_folder.glob("*/*.parquet")))
        df = pd.concat(l)
        df = postprocess(df)

        if save_as is not None:
            if not save_as.parent.exists():
                save_as.parent.mkdir(parents=True)

            print(f"saving: {save_as}")
            df.to_parquet(save_as)

    return df


def cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-folder",
        "-i",
        required=True,
        type=pathlib.Path,
        help="Root folder of the dataset",
    )
    parser.add_argument(
        "--output-folder",
        "-o",
        default="./parquet",
        type=pathlib.Path,
        help="Output folder where to store the postprocessed parquet (default: %(default)s)",
    )
    parser.add_argument(
        "--num-workers",
        "-w",
        default=30,
        type=int,
        help="Number of parallel workers to use for conversion (default: %(default)s)",
    )
    return parser


if __name__ == "__main__":
    args = cli_parser().parse_args()

    main(args.input_folder, args.output_folder / "mirage19.parquet", args.workers)
