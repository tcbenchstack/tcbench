import pandas as pd
import numpy as np

import argparse
import pathlib
import tempfile

from tcbench.libtcdatasets import mirage19_json_to_parquet


def postprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Process the loaded MIRAGE JSON by
    (1) adding a background class;
    (2) adding an "app" column with label information, and encoding it as pandas category
    """
    df = df.assign(
        app=np.where(
            df["android_name"] == df["flow_metadata_bf_label"],
            df["android_name"],
            "background",
        )
    )
    df = df.assign(
        app=np.where(
            df["flow_metadata_bf_activity"] == "Unknown", "background", df["app"]
        )
    )
    df = df.assign(
        app=df["app"].astype("category"),
        packets=df["packet_data_l4_payload_bytes"].apply(len),
    )
    return df


def main(args: argparse.Namespace) -> None:
    if (args.input_folder / "MIRAGE-COVID-CCMA-2022").exists():
        args.input_folder = args.input_folder / "MIRAGE-COVID-CCMA-2022" / "Raw_JSON"

    df = mirage19_json_to_parquet.main(
        args.input_folder, save_as=None, workers=args.num_workers
    )
    df = postprocess(df)

    fname = args.output_folder / "mirage22.parquet"
    if not fname.parent.exists():
        fname.parent.mkdir(parents=True)
    print(f"saving: {fname}")
    df.to_parquet(fname)


def cli_parser():
    return mirage19_json_to_parquet.cli_parser()

if __name__ == "__main__":
    args = cli_parser().parse_args()
    main(args)
