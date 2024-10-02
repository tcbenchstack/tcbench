from __future__ import annotations

from collections import deque, OrderedDict
from typing import Dict, Any

import pathlib
import json
import multiprocessing
import tempfile
import functools
import math
import ipaddress

from multiprocessing import get_context

import polars as pl
import numpy as np

from tcbench import fileutils
from tcbench.cli import richutils
from tcbench.libtcdatasets import curation
from tcbench.libtcdatasets.core import (
    Dataset,
    SequentialPipeStage,
    SequentialPipe,
    DatasetSchema
)
from tcbench.libtcdatasets.constants import (
    DATASET_NAME,
    DATASET_TYPE,
    DATASETS_RESOURCES_FOLDER,
    APP_LABEL_BACKGROUND,
    APP_LABEL_ALL,
)

_POLAR_SCHEMA_RAW = OrderedDict(
    (
        ("packet_data_src_port", pl.List(pl.UInt32())),
        ("packet_data_dst_port", pl.List(pl.UInt32())),
        ("packet_data_packet_dir", pl.List(pl.UInt8())),
        ("packet_data_L4_payload_bytes", pl.List(pl.UInt16())),
        ("packet_data_iat", pl.List(pl.Float32())),
        ("packet_data_TCP_win_size", pl.List(pl.UInt16())),
        ("packet_data_L4_raw_payload", pl.List(pl.List(pl.UInt8()))),
        ("flow_metadata_BF_label", pl.String()),
        ("flow_metadata_BF_labeling_type", pl.String()),
        ("flow_metadata_BF_num_packets", pl.UInt64()),
        ("flow_metadata_BF_IP_packet_bytes", pl.UInt64()),
        ("flow_metadata_BF_L4_payload_bytes", pl.UInt64()),
        ("flow_metadata_BF_duration", pl.Float32()),
        ("flow_metadata_UF_num_packets", pl.UInt64()),
        ("flow_metadata_UF_IP_packet_bytes", pl.UInt64()),
        ("flow_metadata_UF_L4_payload_bytes", pl.UInt64()),
        ("flow_metadata_UF_duration", pl.Float64()),
        ("flow_metadata_DF_num_packets", pl.Int64()),
        ("flow_metadata_DF_IP_packet_bytes", pl.Int64()),
        ("flow_metadata_DF_L4_payload_bytes", pl.Int64()),
        ("flow_metadata_DF_duration", pl.Float64()),
        ("flow_features_packet_length_biflow_min", pl.Float64()),
        ("flow_features_packet_length_biflow_max", pl.Float64()),
        ("flow_features_packet_length_biflow_mean", pl.Float64()),
        ("flow_features_packet_length_biflow_std", pl.Float64()),
        ("flow_features_packet_length_biflow_var", pl.Float64()),
        ("flow_features_packet_length_biflow_mad", pl.Float64()),
        ("flow_features_packet_length_biflow_skew", pl.Float64()),
        ("flow_features_packet_length_biflow_kurtosis", pl.Float64()),
        ("flow_features_packet_length_biflow_10_percentile", pl.Float64()),
        ("flow_features_packet_length_biflow_20_percentile", pl.Float64()),
        ("flow_features_packet_length_biflow_30_percentile", pl.Float64()),
        ("flow_features_packet_length_biflow_40_percentile", pl.Float64()),
        ("flow_features_packet_length_biflow_50_percentile", pl.Float64()),
        ("flow_features_packet_length_biflow_60_percentile", pl.Float64()),
        ("flow_features_packet_length_biflow_70_percentile", pl.Float64()),
        ("flow_features_packet_length_biflow_80_percentile", pl.Float64()),
        ("flow_features_packet_length_biflow_90_percentile", pl.Float64()),
        ("flow_features_packet_length_upstream_flow_min", pl.Float64()),
        ("flow_features_packet_length_upstream_flow_max", pl.Float64()),
        ("flow_features_packet_length_upstream_flow_mean", pl.Float64()),
        ("flow_features_packet_length_upstream_flow_std", pl.Float64()),
        ("flow_features_packet_length_upstream_flow_var", pl.Float64()),
        ("flow_features_packet_length_upstream_flow_mad", pl.Float64()),
        ("flow_features_packet_length_upstream_flow_skew", pl.Float64()),
        ("flow_features_packet_length_upstream_flow_kurtosis", pl.Float64()),
        ("flow_features_packet_length_upstream_flow_10_percentile", pl.Float64()),
        ("flow_features_packet_length_upstream_flow_20_percentile", pl.Float64()),
        ("flow_features_packet_length_upstream_flow_30_percentile", pl.Float64()),
        ("flow_features_packet_length_upstream_flow_40_percentile", pl.Float64()),
        ("flow_features_packet_length_upstream_flow_50_percentile", pl.Float64()),
        ("flow_features_packet_length_upstream_flow_60_percentile", pl.Float64()),
        ("flow_features_packet_length_upstream_flow_70_percentile", pl.Float64()),
        ("flow_features_packet_length_upstream_flow_80_percentile", pl.Float64()),
        ("flow_features_packet_length_upstream_flow_90_percentile", pl.Float64()),
        ("flow_features_packet_length_downstream_flow_min", pl.Float64()),
        ("flow_features_packet_length_downstream_flow_max", pl.Float64()),
        ("flow_features_packet_length_downstream_flow_mean", pl.Float64()),
        ("flow_features_packet_length_downstream_flow_std", pl.Float64()),
        ("flow_features_packet_length_downstream_flow_var", pl.Float64()),
        ("flow_features_packet_length_downstream_flow_mad", pl.Float64()),
        ("flow_features_packet_length_downstream_flow_skew", pl.Float64()),
        ("flow_features_packet_length_downstream_flow_kurtosis", pl.Float64()),
        ("flow_features_packet_length_downstream_flow_10_percentile", pl.Float64()),
        ("flow_features_packet_length_downstream_flow_20_percentile", pl.Float64()),
        ("flow_features_packet_length_downstream_flow_30_percentile", pl.Float64()),
        ("flow_features_packet_length_downstream_flow_40_percentile", pl.Float64()),
        ("flow_features_packet_length_downstream_flow_50_percentile", pl.Float64()),
        ("flow_features_packet_length_downstream_flow_60_percentile", pl.Float64()),
        ("flow_features_packet_length_downstream_flow_70_percentile", pl.Float64()),
        ("flow_features_packet_length_downstream_flow_80_percentile", pl.Float64()),
        ("flow_features_packet_length_downstream_flow_90_percentile", pl.Float64()),
        ("flow_features_iat_biflow_min", pl.Float64()),
        ("flow_features_iat_biflow_max", pl.Float64()),
        ("flow_features_iat_biflow_mean", pl.Float64()),
        ("flow_features_iat_biflow_std", pl.Float64()),
        ("flow_features_iat_biflow_var", pl.Float64()),
        ("flow_features_iat_biflow_mad", pl.Float64()),
        ("flow_features_iat_biflow_skew", pl.Float64()),
        ("flow_features_iat_biflow_kurtosis", pl.Float64()),
        ("flow_features_iat_biflow_10_percentile", pl.Float64()),
        ("flow_features_iat_biflow_20_percentile", pl.Float64()),
        ("flow_features_iat_biflow_30_percentile", pl.Float64()),
        ("flow_features_iat_biflow_40_percentile", pl.Float64()),
        ("flow_features_iat_biflow_50_percentile", pl.Float64()),
        ("flow_features_iat_biflow_60_percentile", pl.Float64()),
        ("flow_features_iat_biflow_70_percentile", pl.Float64()),
        ("flow_features_iat_biflow_80_percentile", pl.Float64()),
        ("flow_features_iat_biflow_90_percentile", pl.Float64()),
        ("flow_features_iat_upstream_flow_min", pl.Float64()),
        ("flow_features_iat_upstream_flow_max", pl.Float64()),
        ("flow_features_iat_upstream_flow_mean", pl.Float64()),
        ("flow_features_iat_upstream_flow_std", pl.Float64()),
        ("flow_features_iat_upstream_flow_var", pl.Float64()),
        ("flow_features_iat_upstream_flow_mad", pl.Float64()),
        ("flow_features_iat_upstream_flow_skew", pl.Float64()),
        ("flow_features_iat_upstream_flow_kurtosis", pl.Float64()),
        ("flow_features_iat_upstream_flow_10_percentile", pl.Float64()),
        ("flow_features_iat_upstream_flow_20_percentile", pl.Float64()),
        ("flow_features_iat_upstream_flow_30_percentile", pl.Float64()),
        ("flow_features_iat_upstream_flow_40_percentile", pl.Float64()),
        ("flow_features_iat_upstream_flow_50_percentile", pl.Float64()),
        ("flow_features_iat_upstream_flow_60_percentile", pl.Float64()),
        ("flow_features_iat_upstream_flow_70_percentile", pl.Float64()),
        ("flow_features_iat_upstream_flow_80_percentile", pl.Float64()),
        ("flow_features_iat_upstream_flow_90_percentile", pl.Float64()),
        ("flow_features_iat_downstream_flow_min", pl.Float64()),
        ("flow_features_iat_downstream_flow_max", pl.Float64()),
        ("flow_features_iat_downstream_flow_mean", pl.Float64()),
        ("flow_features_iat_downstream_flow_std", pl.Float64()),
        ("flow_features_iat_downstream_flow_var", pl.Float64()),
        ("flow_features_iat_downstream_flow_mad", pl.Float64()),
        ("flow_features_iat_downstream_flow_skew", pl.Float64()),
        ("flow_features_iat_downstream_flow_kurtosis", pl.Float64()),
        ("flow_features_iat_downstream_flow_10_percentile", pl.Float64()),
        ("flow_features_iat_downstream_flow_20_percentile", pl.Float64()),
        ("flow_features_iat_downstream_flow_30_percentile", pl.Float64()),
        ("flow_features_iat_downstream_flow_40_percentile", pl.Float64()),
        ("flow_features_iat_downstream_flow_50_percentile", pl.Float64()),
        ("flow_features_iat_downstream_flow_60_percentile", pl.Float64()),
        ("flow_features_iat_downstream_flow_70_percentile", pl.Float64()),
        ("flow_features_iat_downstream_flow_80_percentile", pl.Float64()),
        ("flow_features_iat_downstream_flow_90_percentile", pl.Float64()),
    )
)

_POLAR_SCHEMA_PREPROCESS = _POLAR_SCHEMA_RAW.copy()
_POLAR_SCHEMA_PREPROCESS.update(
    OrderedDict(
        (
            ("src_ip", pl.String()),
            ("src_port", pl.Int32()),
            ("dst_ip", pl.String()),
            ("dst_port", pl.Int32()),
            ("proto_id", pl.Int32()),
            ("device_id", pl.String()),
            ("fname", pl.String()),
            ("fname_row_idx", pl.Int64()),
        )
    )
)


def _reformat_json_entry(
    json_entry: Dict[str, Any], 
    fields_order: List[str]
) -> Dict[str, Any]:
    """Process a JSON nested structure by chaining partial names via "_" """
    data = OrderedDict()
    for field_name in fields_order:
        data[field_name] = None

    queue = deque(json_entry.items())
    while len(queue):
        key, value = queue.popleft()
        if not isinstance(value, dict):
            if not isinstance(value, (list, str)) and math.isnan(value):
                value = "NaN"
            data[key] = value
            continue
        for inner_key, inner_value in value.items():
            queue.append((f"{key}_{inner_key}", inner_value))
    return data


def _json_entry_to_dataframe(
    json_entry: Dict[str, Any], 
    dset_schema: DatasetSchema
) -> pl.DataFrame:
    """Create a DataFrame by flattening the JSON nested structure
    chaining partial names via "_"
    """
    json_entry = _reformat_json_entry(json_entry, dset_schema.fields)
    for key, value in json_entry.items():
        if value == "NaN":
            value = np.nan
        # Note: enforce values to be list for pl.DataFrame conversion
        json_entry[key] = [value]
    return pl.DataFrame(json_entry, schema=dset_schema.to_polars())


def _load_raw_json_worker(
    fname: pathlib.Path, 
    dset_schema: DatasetSchema, 
    save_to: pathlib.Path,
) -> None:
    fname = pathlib.Path(fname)
    with open(fname) as fin:
        data = json.load(fin)

    with open(save_to / f"{fname.parent.name}__{fname.name}", "w") as fout:
        for idx, (flow_id, json_entry) in enumerate(data.items()):
            # adding a few extra columns after parsing raw data
            json_entry = _reformat_json_entry(json_entry, dset_schema.fields)
            src_ip, src_port, dst_ip, dst_port, proto_id = \
                flow_id.split(",")
            json_entry["src_ip"] = src_ip
            json_entry["src_port"] = int(src_port)
            json_entry["dst_ip"] = dst_ip
            json_entry["dst_port"] = int(dst_port)
            json_entry["proto_id"] = int(proto_id)
            json_entry["device_id"] = fname.parent.name
            json_entry["fname"] = fname.stem
            json_entry["fname_row_idx"] = idx
            json.dump(json_entry, fout)
            fout.write("\n")

def load_raw_json(fname: pathlib.Path, dataset_name: DATASET_NAME) -> pl.DataFrame:
    import tcbench
    fname = pathlib.Path(fname)
    with open(fname) as fin:
        data = json.load(fin)

    dset_schema = (
        tcbench
        .datasets_catalog()
        [dataset_name]
        .get_schema(DATASET_TYPE.RAW)
    )

    l = []
    for idx, (flow_id, json_entry) in enumerate(data.items()):
        json_entry = _reformat_json_entry(json_entry, dset_schema.fields)
        src_ip, src_port, dst_ip, dst_port, proto_id = \
            flow_id.split(",")
        json_entry["src_ip"] = src_ip
        json_entry["src_port"] = int(src_port)
        json_entry["dst_ip"] = dst_ip
        json_entry["dst_port"] = int(dst_port)
        json_entry["proto_id"] = int(proto_id)
        json_entry["device_id"] = fname.parent.name
        json_entry["fname"] = fname.stem
        json_entry["fname_row_idx"] = idx
        l.append(_json_entry_to_dataframe(json_entry, dset_schema))
    return pl.concat(l)

def _rename_columns(columns: List[str]) -> Dict[str, str]:
    rename = dict()
    for col in columns:
        new_name = col.lower()
        if col.startswith("packet_data_"):
            new_name = (
                new_name.replace("packet_data", "pkts")
                .replace("packet_dir", "dir")
                .replace("l4_payload_bytes", "size")
                .replace("l4_raw_payload", "raw_payload")
            )
        elif col.startswith("flow_metadata"):
            new_name = (
                new_name.replace("flow_metadata_", "")
                .replace("bf_", "")
                .replace("num_packets", "packets")
                .replace("ip_packet_bytes", "bytes")
                .replace("l4_payload_bytes", "bytes_payload")
            )
            if "uf" in new_name:
                new_name = new_name.replace("uf_", "") + "_upload"
            elif "df" in new_name:
                new_name = new_name.replace("df_", "") + "_download"
        elif col.startswith("flow_features_"): 
            new_name = (
                new_name
                .replace("flow_features_", "")
                .replace("packet_length", "packet_size")
                .replace("_biflow", "")
            ) 
            if new_name.endswith("percentile"):
                _1, q, _2 = new_name.rsplit("_", 2)
                new_name = new_name.replace(f"{q}_percentile", f"q{q}")
            if "upstream_flow" in new_name:
                new_name = new_name.replace("_upstream_flow", "") + "_upload"
            elif "downstream_flow" in new_name:
                new_name = new_name.replace("_downstream_flow", "") + "_download"
        rename[col] = new_name
    return rename


class Mirage19(Dataset):
    def __init__(self):
        super().__init__(name=DATASET_NAME.MIRAGE19)
        self.df_app_metadata = pl.read_csv(
            DATASETS_RESOURCES_FOLDER / f"{self.name}_app_metadata.csv"
        )

    @property
    def _list_raw_json_files(self):
        return list(self.folder_raw.rglob("*.json"))

    def _parse_raw_json(
        self, 
    ) -> pl.DataFrame:
        files = self._list_raw_json_files
        dset_schema = self.get_schema(DATASET_TYPE.RAW)

        with tempfile.TemporaryDirectory() as tmp_folder:
            tmp_folder = pathlib.Path(tmp_folder)
            func = functools.partial(
                _load_raw_json_worker, 
                dset_schema=dset_schema, 
                save_to=tmp_folder,
            )
            with (
                richutils.Progress(
                    description="Parse JSON files...", 
                    total=len(files)
                ) as progress,
                multiprocessing.Pool(processes=2) as pool,
            ):
                for _ in pool.imap_unordered(func, files):
                    progress.update()

            with richutils.SpinnerProgress(description="Reload..."):
                df = (
                    pl.read_ndjson(
                        tmp_folder, 
                        schema=dset_schema.to_polars()
                    )
                    .sort(
                        "device_id", 
                        "fname", 
                        "fname_row_idx"
                    )
                )
        return df

    def raw(self) -> pl.DataFrame:
        df = self._parse_raw_json()
        with richutils.SpinnerProgress(description="Writing parquet files..."):
            fileutils.save_parquet(
                df, 
                save_as=self.folder_raw/f"{self.name}.parquet", 
                echo=False
            )
        return df

    def _raw_postprocess_rename_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.rename(_rename_columns(df.columns))

    def _raw_postprocess_add_app_and_background(self, df: pl.DataFrame) -> pl.DataFrame:
        return (
            df
            # add app column using static metadata
            .join(
                self.df_app_metadata,
                left_on="label",
                right_on="android_package_name",
                how="left",
            )
            .with_columns(
                # flows without a recognized label are re-labeled as background
                app=(pl.col("app").fill_null(APP_LABEL_BACKGROUND))
            )
            .with_columns(
                # force to background flows with UDP packets of size zero
                app=(
                    pl.when(
                        (pl.col("proto") == "udp").and_(
                            pl.col("pkts_size").list.min() == 0
                        )
                    )
                    .then(pl.lit(APP_LABEL_BACKGROUND))
                    .otherwise(pl.col("app"))
                )
            )
        )

    def _raw_postprocess_add_other_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        # add column: convert proto_id to string (6->tcp, 17->udp)
        df = df.with_columns(
            proto=(
                pl.when(pl.col("proto_id").eq(6))
                .then(pl.lit("tcp"))
                .otherwise(pl.lit("udp"))
            ),
        )

        # add columns: ip addresses private/public
        df = curation.add_is_private_ip_columns(df)
        # add columns: check if tcp handshake is valid
        df = curation.add_is_valid_tcp_handshake(
            df, tcp_handshake_size=0, direction_upload=0, direction_download=1
        )

        return (
            df
            # add a global row_id
            .with_row_index(name="row_id")
        )


    def _raw_postprocess(self) -> pl.DataFrame:
        def _get_stats(df):
            df_stats = curation.get_stats(df)
            return (df, df_stats)

        def _write_parquet_files(tpl):
            df, df_stats = tpl
            df.write_parquet(
                self.folder_raw / "_postprocess.parquet"
            )
            df_stats.write_parquet(
                self.folder_raw / f"_postprocess_stats.parquet"
            )
            return df, df_stats

        # attempt at loading the previously generate raw version
        df = fileutils.load_if_exists(self.folder_raw / f"{self.name}.parquet", echo=False)
        if df is None:
            df = self.raw()
        # ...and triggering postprocessing steps        
        df, _ = SequentialPipe(
            SequentialPipeStage(
                self._raw_postprocess_rename_columns,
                name="Rename columns",
            ),
            SequentialPipeStage(
                self._raw_postprocess_add_other_columns,
                name="Add columns", 
            ),
            SequentialPipeStage(
                self._raw_postprocess_add_app_and_background,
                name="Add metadata",
            ),
            SequentialPipeStage(
                _get_stats,
                name="Compute statistics",
            ),
            SequentialPipeStage(
                _write_parquet_files,
                name="Write parquet files",
            ),
            name="Postprocess raw..."
        ).run(df) 

        return df

    def _curate_rename(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.rename({
            "label": "android_package_name",
        })

    def _curate_drop_background(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            # force to background flows with UDP packets of size zero
            app=(
                pl.when(
                    (pl.col("proto") == "udp").and_(pl.col("pkts_size").list.min() == 0)
                )
                .then(pl.lit(APP_LABEL_BACKGROUND))
                .otherwise(pl.col("app"))
            )
        ).filter(pl.col("app") != APP_LABEL_BACKGROUND)

    def _curate_adjust_packet_series(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            # increase packets size to reflect the expected true size
            # for TCP, add 40 bytes
            # for UDP, add 28 bytes
            pkts_size=(
                pl.when(pl.col("proto") == "tcp")
                .then(pl.col("pkts_size").list.eval(pl.element() + 40))
                .otherwise(pl.col("pkts_size").list.eval(pl.element() + 28))
            ),
            # enforce direction (0/upload: 1, 1/download: -1)
            pkts_dir=(
                pl.col("pkts_dir").list.eval(
                    pl.when(pl.element() == 0).then(1).otherwise(-1)
                )
            ),
        )

    def _curate_add_pkt_indices_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            # series with the index of TCP acks packets
            pkts_ack_idx=(
                pl.when(pl.col("proto") == "tcp")
                # for TCP, acks are enforced to 40 bytes
                .then(curation.expr_pkts_ack_idx(ack_size=40))
                # for UDP, packets are always larger then 0 bytes
                # so the following is selecting all indices
                .otherwise(curation.expr_pkts_ack_idx(ack_size=0))
            ),
            # series with the index of data packets
            pkts_data_idx=(
                pl.when(pl.col("proto") == "tcp")
                # for TCP, acks are enforced to 40 bytes
                .then(curation.expr_pkts_data_idx(ack_size=40))
                # for UDP, packets are always larger then 0 bytes
                # so the following is selecting all indices
                .otherwise(curation.expr_pkts_data_idx(ack_size=0))
            ),
        )

    def _curate_add_other_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return (df
            .with_columns(
                # length of all series
                pkts_len=(pl.col("pkts_size").list.len()),
                # flag to indicate if the packet sizes have all packets
                pkts_is_complete=(pl.col("pkts_size").list.len() == pl.col("packets")),
                # series pkts_size * pkts_dir
                pkts_size_times_dir=(curation.expr_pkts_size_times_dir()),
            )
            .with_columns(
                # number of ack packets
                packets_ack=(pl.col("pkts_ack_idx").list.len()),
                # number of ack packets in upload
                packets_ack_upload=(
                    curation.expr_list_len_upload("pkts_size_times_dir", "pkts_ack_idx")
                ),
                # number of ack packets in download
                packets_ack_download=(
                    curation.expr_list_len_download("pkts_size_times_dir", "pkts_ack_idx")
                ),
                # number of data packets
                packets_data=(pl.col("pkts_data_idx").list.len()),
                # number of ack packets in upload
                packets_data_upload=(
                    curation.expr_list_len_upload("pkts_size_times_dir", "pkts_data_idx")
                ),
                # number of ack packets in download
                packets_data_download=(
                    curation.expr_list_len_download("pkts_size_times_dir", "pkts_data_idx")
                ),
            )
        )

    def _curate_drop_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.drop(
            [
                "pkts_src_port",
                "pkts_dst_port",
                "pkts_raw_payload",
                "labeling_type",
            ]
        )

    def _curate_final_filter(self, df: pl.DataFrame, min_pkts: int = None) -> pl.DataFrame:
        df_new = df.filter(
            # flows starting with a complete handshake
            pl.col("is_valid_handshake")
        )

        if min_pkts is not None:
            # flows with at least a specified number of packets
            df_new = df.filter(pl.col("packets") >= min_pkts)
        return df_new

    def curate(self) -> pl.DataFrame:
        def _get_stats(df):
            df_stats = curation.get_stats(df)
            return (df, df_stats)

        def _get_splits(tpl):
            df, df_stats = tpl
            self.df = df
            df_splits = self.compute_splits(
                num_splits=10,
                test_size=0.1,
                seed=1,
            )
            self.df = None
            return (df, df_stats, df_splits)

        def _write_parquet_files(tpl):
            df, df_stats, df_splits = tpl
            folder = self.folder_curate
            if not folder.exists():
                folder.mkdir(parents=True)
            df.write_parquet(folder / f"{self.name}.parquet")
            df_stats.write_parquet(
                folder / f"{self.name}_stats.parquet"
            )
            df_splits.write_parquet(
                folder / f"{self.name}_splits.parquet"
            )
            return df, df_stats, df_splits

        df = self._raw_postprocess() 

        self.df, self.df_stats, self.df_splits = SequentialPipe(
            SequentialPipeStage(
                self._curate_rename,
                name="Column renaming",
            ),
            SequentialPipeStage(
                self._curate_drop_background, 
                name="Drop background flows"
            ),
            SequentialPipeStage(
                self._curate_adjust_packet_series,
                name="Adjust packet series",
            ),
            SequentialPipeStage(
                self._curate_add_pkt_indices_columns,
                name="Add packet series indices"
            ),
            SequentialPipeStage(
                self._curate_add_other_columns,
                name="Add more columns",
            ),
            SequentialPipeStage(
                self._curate_drop_columns,
                name="Drop columns",
            ),
            SequentialPipeStage(
                self._curate_final_filter,
                name="Filter out flows",
            ),
            SequentialPipeStage(
                _get_stats,
                name="Compute statistics",
            ),
            SequentialPipeStage(
                _get_splits,
                name="Compute splits",
            ),
            SequentialPipeStage(
                _write_parquet_files,
                name="Write parquet files",
            ),
            name="Curation..."
        ).run(df)

        return self.df

class Mirage22(Mirage19):
    def __init__(self):
        super(Mirage19, self).__init__(name=DATASET_NAME.MIRAGE22)
        self.df_app_metadata = pl.read_csv(
            DATASETS_RESOURCES_FOLDER / f"{self.name}_app_metadata.csv"
        )

    @property
    def _list_raw_json(self) -> List[pathlib.Path]:
        return list(
            (
                self.folder_raw 
                / "MIRAGE-COVID-CCMA-2022" 
                / "Raw_JSON"
            ).rglob("*.json")
        )

    def install(self, no_download: bool = False) -> pathlib.Path:
        subfolder = (
            self.install_folder 
            / "raw" 
            / "MIRAGE-COVID-CCMA-2022" 
            / "Raw_JSON"
        )
        extra_unpack = (
            subfolder / "Discord.zip",
            subfolder / "Meet.zip",
            subfolder / "Slack.zip",
            subfolder / "Zoom.zip",
            subfolder / "GotoMeeting.zip",
            subfolder / "Messenger.zip",
            subfolder / "Teams.zip",
            subfolder / "Skype.zip",
            subfolder / "Webex.zip",
        )
        return super().install(no_download, extra_unpack)

    def preprocess(self):
        pass

    def curate(self):
        pass
