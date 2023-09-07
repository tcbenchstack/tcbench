import pandas as pd
import numpy as np
import dask.dataframe as dd
import pyarrow as pa
import rich.progress as richprogress

from distributed import Client, progress

import shutil
import tempfile
import multiprocessing
import pathlib
import argparse
import dateutil
import functools

COLUMNS_TO_KEEP = [
    "frame_time",
    "ip_src",
    "ip_dst",
    "ip_proto",
    "tcp_srcport",
    "tcp_dstport",
    "udp_srcport",
    "udp_dstport",
    "tcp_len",
    "udp_length",
    "partition",
    "fname",
    "location",
]

from tcbench import cli

console = cli.console


def create_connection_id(ser: pd.Series) -> str:
    """Associate a 5-tuple connection id to a flow

    Arguments:
        ser: a pandas Series with per-flow data

    Return:
        A string encoding the 5-tuple connection id
    """
    proto = ser["ip_proto"]
    src_ip, dst_ip = ser[["ip_src", "ip_dst"]].values
    if proto == 6:
        src_port, dst_port = ser[["tcp_srcport", "tcp_dstport"]].values
    else:
        src_port, dst_port = ser[["udp_srcport", "udp_dstport"]].values

    if src_ip > dst_ip:
        src_ip, src_port, dst_ip, dst_port = dst_ip, dst_port, src_ip, src_port
    conn_id = f"{src_ip}_{src_port}_{dst_ip}_{dst_port}_{proto}"
    return conn_id

INVALID_IPS = {
    "127.0.0.1",
    "0.0.0.0",
    "255.255.255.255",
}
def worker(fname: str, tmp_folder: str) -> None:
    """Helper function responsible for processing a
    utmobilenet21 CSV and save it into a temporary folder

    Arguments:
        fname: the input CSV to process
        tmp_folder: the temporary folder where to store
            intermediate output
    """
    # load everything as string (type casting moved later)
    df = pd.read_csv(fname, dtype=str)
    df = df.assign(
        fname=fname.name,
        partition=fname.parent.name,
    )

    # reformat columns name
    df.columns = [col.replace(".", "_") for col in df.columns]

    if "location" not in df.columns:
        df = df.assign(location="")

    # drop columns not needed
    df = df[COLUMNS_TO_KEEP]

    # keep only TCP and UDP
    df = df[
        (df["ip_proto"].isin({"6", "17", "6.0", "17.0"}))
        & (~df["ip_src"].isna())
        & (~df["ip_dst"].isna())
        & (~df["ip_src"].isin(INVALID_IPS))
        & (~df["ip_dst"].isin(INVALID_IPS))
    ]

    # extract app label
    func_parse_timestamp = functools.partial(
        dateutil.parser.parse, tzinfos={"CDT": -5 * 3600}
    )
    df = df.assign(
        app=df["fname"].apply(lambda text: text.split(" ", 1)[0]),
        frame_time=df["frame_time"]
        .apply(lambda text: float(func_parse_timestamp(text).strftime("%s.%f")))
        .astype(float),
        conn_id=df.apply(create_connection_id, axis=1).astype(str),
    ).to_parquet(tmp_folder / fname.with_suffix(".parquet").name)

    # print(".", end="", flush=True)
    # return df

    # ddf = dd.from_pandas(df, npartitions=1)

    # print(".", end="", flush=True)
    # return ddf


def create_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """Compose flow time series by grouping packets the same flow into
    numpy arrays

    Argument:
        df: the input pandas DataFrame where each entry corresponds to a different packet

    Return:
        The generate per-flow pandas DataFrame containing the following
        columns ("src_ip", "src_port", "dst_ip", "dst_port", "ip_proto",
        "first", "last", "duration", "packets", "bytes", "timetofirst",
        "pkts_size", "pkts_dir", "partition", "location", "fname", "app"
        )
    """
    if df.shape[0] == 0:
        return pd.DataFrame(
            columns=[
                "src_ip",
                "src_port",
                "dst_ip",
                "dst_port",
                "ip_proto",
                "first",
                "last",
                "duration",
                "packets",
                "bytes",
                "timetofirst",
                "pkts_size",
                "pkts_dir",
                "partition",
                "location",
                "fname",
                "app",
            ]
        )

    df_tmp = df.sort_values(by="frame_time")

    first_pkt = df_tmp.iloc[0]

    src_ip, dst_ip = first_pkt["ip_src"], first_pkt["ip_dst"]
    first = first_pkt["frame_time"]
    last = df_tmp.iloc[-1]["frame_time"]
    duration = last - first
    packets = df_tmp.shape[0]
    ip_proto = first_pkt["ip_proto"]
    partition = first_pkt["partition"]
    location = first_pkt["location"]
    fname = first_pkt["fname"]
    app = fname.split("_", 1)[0]

    if df_tmp["ip_proto"].values[0] == 6:
        pkts_size = df_tmp["tcp_len"].fillna(0).values
        src_port, dst_port = first_pkt["tcp_srcport"], first_pkt["tcp_dstport"]
    else:
        pkts_size = df_tmp["udp_length"].fillna(0).values
        src_port, dst_port = first_pkt["udp_srcport"], first_pkt["udp_dstport"]
    _bytes = pkts_size.sum()
    timetofirst = df_tmp["frame_time"].diff().fillna(0).values

    pkts_dir = (df_tmp["ip_src"] == first_pkt["ip_src"]).astype(int).values

    df_res = pd.DataFrame(
        [
            [
                src_ip,
                src_port,
                dst_ip,
                dst_port,
                ip_proto,
                first,
                last,
                duration,
                packets,
                _bytes,
                timetofirst,
                pkts_size,
                pkts_dir,
                partition,
                location,
                fname,
                app,
            ]
        ],
        columns=[
            "src_ip",
            "src_port",
            "dst_ip",
            "dst_port",
            "ip_proto",
            "first",
            "last",
            "duration",
            "packets",
            "bytes",
            "timetofirst",
            "pkts_size",
            "pkts_dir",
            "partition",
            "location",
            "fname",
            "app",
        ],
    )
    return df_res


def main(args: argparse.Namespace):
    """Main function loading CSVs and converting them into
    a monolithic parquet file
    """
    if (args.input_folder / "csvs").exists():
        args.input_folder /= "csvs"

    with Client(n_workers=args.num_workers) as client:
        staging_folder = pathlib.Path(args.tmp_staging_folder)
        output_folder = pathlib.Path(args.output_folder)

        extra_line = False
        for partition in args.input_folder.iterdir():
            if extra_line:
                print()
            print(f"processing: {partition}")
            extra_line = True

            files = list(partition.glob("*.csv"))

            partition_name = partition.name.lower().replace(" ", "_")
            curr_staging_folder = staging_folder / partition_name

            if curr_staging_folder.exists():
                shutil.rmtree(str(curr_staging_folder))
            curr_staging_folder.mkdir(parents=True)

            ##############
            # stage1: convert csv to parquet + some cleaning
            ##############
            (curr_staging_folder / "stage1").mkdir(parents=True)
            func = functools.partial(worker, tmp_folder=curr_staging_folder / "stage1")
            print(f"found {len(files)} files")
            with richprogress.Progress(
                richprogress.TextColumn("[progress.description]{task.description}"),
                richprogress.BarColumn(),
                richprogress.MofNCompleteColumn(),
                richprogress.TimeElapsedColumn(),
                console=console,
            ) as progressbar:
                task_id = progressbar.add_task("Converting CSVs...", total=len(files))
                with multiprocessing.Pool(min(len(files), 30)) as pool:
                    for item in pool.imap_unordered(func, files):
                        progressbar.advance(task_id)
            print("stage1 completed")

            #########################
            # stage2: repartition (if needed)
            #########################
            ddf = dd.read_parquet(curr_staging_folder / "stage1")
            if len(files) > 1000:
                ddf = ddf.reset_index(drop=True).repartition(50)
            ddf.reset_index(drop=True).persist()
            progress(ddf)
            ddf.to_parquet(curr_staging_folder / "stage2")
            print("stage2 completed")

            #########################
            # stage3: minor types conversion
            #########################
            ddf = dd.read_parquet(curr_staging_folder / "stage2").reset_index(drop=True)
            # convert all to float (because some numbers are float?)
            # then convert to int64 (which supports nan)
            ddf1 = ddf.astype(
                {
                    "frame_time": float,
                    "ip_proto": float,
                    "tcp_len": float,
                    "tcp_srcport": float,
                    "tcp_dstport": float,
                    "udp_srcport": float,
                    "udp_dstport": float,
                    "udp_length": float,
                }
            ).persist()
            progress(ddf1)
            ddf1.to_parquet(curr_staging_folder / "stage3")  # , schema=pa_schema)
            print("stage3 completed")

            #########################
            # last: time series creation
            #########################
            ddf = dd.read_parquet(curr_staging_folder / "stage3").reset_index(drop=True)

            meta = pd.DataFrame(
                dtype=object,
                columns=[
                    "src_ip",
                    "src_port",
                    "dst_ip",
                    "dst_port",
                    "ip_proto",
                    "first",
                    "last",
                    "duration",
                    "packets",
                    "bytes",
                    "timetofirst",
                    "pkts_size",
                    "pkts_dir",
                    "partition",
                    "location",
                    "fname",
                    "app",
                ],
            )
            meta = meta.astype(
                {
                    "src_ip": str,
                    "dst_ip": str,
                    "src_port": int,
                    "dst_port": int,
                    "ip_proto": int,
                    "first": float,
                    "last": float,
                    "duration": float,
                    "packets": int,
                    "bytes": int,
                    "partition": str,
                    "location": str,
                    "fname": str,
                    "app": str,
                }
            )
            schema = dict(meta.dtypes)
            pa_schema = pa.schema(
                [
                    (
                        name.lower(),
                        pa.string()
                        if dtype == np.dtype(object)
                        else pa.from_numpy_dtype(dtype),
                    )
                    for name, dtype in schema.items()
                    if name not in ("timetofirst", "pkts_size", "pkts_dir")
                ]
            )
            pa_schema = pa_schema.append(pa.field("pkts_size", pa.list_(pa.int64())))
            pa_schema = pa_schema.append(pa.field("pkts_dir", pa.list_(pa.int64())))
            pa_schema = pa_schema.append(
                pa.field("timetofirst", pa.list_(pa.float64()))
            )

            ddf2 = ddf.groupby("conn_id").apply(create_time_series, meta=meta).persist()
            progress(ddf2)
            ddf2.reset_index(drop=True).to_parquet(
                curr_staging_folder / "stage4", schema=pa_schema
            )
            print("stage4 completed")

            ### we can finally pack everything together
            df = (
                dd.read_parquet(curr_staging_folder / "stage4")
                .reset_index(drop=True)
                .compute()
            )
            fname = (staging_folder / partition_name).with_suffix(".parquet")
            if not fname.parent.exists():
                fname.parent.mkdir(parents=True)
            print(f"saving: {fname}")
            df.to_parquet(fname)

        print("merging all partitions")
        if not output_folder.exists():
            output_folder.mkdir(parents=True)
        df = dd.read_parquet(staging_folder / "*.parquet").compute()
        df = df.reset_index(drop=True).reset_index().rename({"index": "row_id"}, axis=1)
        df = df.assign(app=df["app"].astype("category"))

        fname = output_folder / "utmobilenet21.parquet"
        print(f"saving: {fname}")
        df.to_parquet(fname)


def cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", type=pathlib.Path)
    parser.add_argument(
        "--output-folder",
        type=pathlib.Path,
        default="./parquet",
        help="Output folder where to store the postprocessed parquet (default: %(default)s)",
    )
    parser.add_argument(
        "--tmp-staging-folder",
        type=pathlib.Path,
        default="/tmp/processing-utmobilenet21",
        help="Temporary folder where to store the intermediate results during conversion (default: %(default)s)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of parallel workers to use for conversion (default: %(default)s)",
    )
    return parser


if __name__ == "__main__":
    args = cli_parser().parse_args()

    main(args)
