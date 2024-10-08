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
    SequentialPipelineStage,
    SequentialPipeline,
    DatasetSchema
)
from tcbench.libtcdatasets.constants import (
    DATASET_NAME,
    DATASET_TYPE,
    DATASETS_RESOURCES_FOLDER,
    APP_LABEL_BACKGROUND,
    APP_LABEL_ALL,
)


def _reformat_json_entry(
    json_entry: Dict[str, Any], 
    fields_order: List[str] = None,
) -> Dict[str, Any]:
    """Process a JSON nested structure by chaining partial names via "_" """
    data = OrderedDict()
    if fields_order is None:
        fields_order = []
    for field_name in fields_order:
        data[field_name] = None

    queue = deque(json_entry.items())
    while len(queue):
        key, value = queue.popleft()
        if not isinstance(value, dict):
            if (
                value is not None 
                and not isinstance(value, (list, str)) 
                and math.isnan(value)
            ):
                value = "null"
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
            json_entry["fname"] = fname.stem
            json_entry["fname_row_idx"] = idx
            if dset_schema.dataset_name == DATASET_NAME.MIRAGE19:
                json_entry["parent_folder"] = fname.parent.name
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
        json_entry["fname"] = fname.stem
        json_entry["fname_row_idx"] = idx
        if dataset_name == DATASET_NAME.MIRAGE19:
            json_entry["parent_folder"] = fname.parent.name
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
                .replace("ip_packet_bytes", "l3_size")
                .replace("ip_header_bytes", "l3_header_size")
                .replace("l4_payload_bytes", "l4_size")
                .replace("l4_header_bytes", "l4_header_size")
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


class BaseParserRawJSON:
    def __init__(self, name: DATASET_NAME, dset_schema: DatasetSchema):
        self.name = name
        self.dset_schema = dset_schema

    def _parse_raw_json(
        self, 
        *files: Iterable[pathlib.Path],
        sort_by: Iterable[str] = None,
    ) -> pl.DataFrame:
        if sort_by is None:
            sort_by = (
                "parent_folder", 
                "fname", 
                "fname_row_idx"
            )

        with tempfile.TemporaryDirectory() as tmp_folder:
            tmp_folder = pathlib.Path(tmp_folder)
            func = functools.partial(
                _load_raw_json_worker, 
                dset_schema=self.dset_schema, 
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
                        schema=self.dset_schema.to_polars()
                    )
                    .sort(*sort_by)
                )
        return df

    def run(
        self, 
        *files: Iterable[pathlib.Path], 
        save_to: pathlib.Path = None,
        sort_by: Iterable[str] = None,
    ) -> pl.DataFrame:
        df = self._parse_raw_json(*files, sort_by=sort_by)
        if save_to is None:
            save_to = pathlib.Path(".")
        with richutils.SpinnerProgress(description="Writing parquet files..."):
            fileutils.save_parquet(
                df, 
                save_as=save_to/f"{self.name}.parquet", 
                echo=False
            )
        return df


class BaseRawPostprocessingPipeline(SequentialPipeline):
    def __init__(
        self, 
        df_app_metadata: pl.DataFrame,
        save_to: pathlib.Path, 
    ):
        self.save_to = save_to
        self.df_app_metadata = df_app_metadata

        stages = [
            SequentialPipelineStage(
                self._rename_columns,
                name="Rename columns",
            ),
            SequentialPipelineStage(
                self._add_other_columns,
                name="Add columns", 
            ),
            SequentialPipelineStage(
                self._add_app_and_background,
                name="Add metadata",
            ),
            SequentialPipelineStage(
                self._compute_stats,
                name="Compute statistics",
            ),
            SequentialPipelineStage(
                self._write_parquet_files,
                name="Write parquet files",
            ),
        ]
        super().__init__(
            *stages, 
            name="Postprocess raw...",
            progress=True
        )

    def _rename_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.rename(_rename_columns(df.columns))

    def _add_app_and_background(self, df: pl.DataFrame) -> pl.DataFrame:
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
                        (pl.col("proto_id") == 17).and_(
                            pl.col("pkts_size").list.min() == 0
                        )
                    )
                    .then(pl.lit(APP_LABEL_BACKGROUND))
                    .otherwise(pl.col("app"))
                )
            )
        )

    def _add_other_columns(self, df: pl.DataFrame) -> pl.DataFrame:
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
        df = curation.add_is_valid_tcp_handshake_heuristic(
            df, tcp_handshake_size=0, direction_upload=0, direction_download=1
        )
        return (
            df
            # add a global row_id
            .with_row_index(name="row_id")
        )

    def _compute_stats(self, df):
        df_stats = curation.get_stats(df)
        return (df, df_stats)

    def _write_parquet_files(self, df, df_stats):
        df.write_parquet(
            self.save_to / "_postprocess.parquet"
        )
        df_stats.write_parquet(
            self.save_to / f"_postprocess_stats.parquet"
        )
        return df, df_stats


class BaseCuratePipeline(SequentialPipeline):
    def __init__(
        self, 
        dataset_name: DATASET_NAME,
        save_to: pathlib.Path,
        dset_schema: DatasetSchema,
    ):
        self.dataset_name = dataset_name
        self.save_to = save_to
        self.dset_schema = dset_schema

        stages = [
            SequentialPipelineStage(
                self._rename_columns,
                name="Rename columns",
            ),
            SequentialPipelineStage(
                self._drop_background, 
                name="Drop background flows"
            ),
            SequentialPipelineStage(
                self._adjust_packet_series,
                name="Adjust packet series",
            ),
            SequentialPipelineStage(
                self._add_pkt_indices_columns,
                name="Add packet series indices"
            ),
            SequentialPipelineStage(
                self._add_other_columns,
                name="Add more columns",
            ),
            SequentialPipelineStage(
                self._drop_columns,
                name="Drop columns",
            ),
            SequentialPipelineStage(
                self._final_filter,
                name="Filter out flows",
            ),
            SequentialPipelineStage(
                self._compute_stats,
                name="Compute statistics",
            ),
            SequentialPipelineStage(
                self._compute_splits,
                name="Compute splits",
            ),
            SequentialPipelineStage(
                self._write_parquet_files,
                name="Write parquet files",
            ),
        ]
        super().__init__(
            *stages, 
            name="Curation...",
            progress=True
        )

    def _rename_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.rename({
            "label": "android_package_name",
        })

    def _drop_background(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("app") != APP_LABEL_BACKGROUND)

    def _adjust_packet_series(self, df: pl.DataFrame) -> pl.DataFrame:
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

    def _add_pkt_indices_columns(self, df: pl.DataFrame) -> pl.DataFrame:
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

    def _add_other_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return (
            df
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

    def _drop_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.drop(
            [
                "pkts_src_port",
                "pkts_dst_port",
                "pkts_raw_payload",
                "labeling_type",
                "proto_id",
            ]
        )

    def _final_filter(self, df: pl.DataFrame, min_pkts: int = None) -> pl.DataFrame:
        df_new = df.filter(
            # flows starting with a complete handshake
            pl.col("is_valid_handshake")
        )

        if min_pkts is not None:
            # flows with at least a specified number of packets
            df_new = df.filter(pl.col("packets") >= min_pkts)
        return df_new

    def _compute_stats(
        self, 
        df: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        df_stats = curation.get_stats(df)
        return (df, df_stats)

    def _compute_splits(
        self, 
        df: pl.DataFrame, 
        df_stats: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        from tcbench.modeling import splitting
        df_splits = splitting.split_monte_carlo(
            df,
            y_colname="app",
            index_colname="row_id", 
            num_splits=10,
            seed=1,
            test_size=0.1,
        )
        return (df, df_stats, df_splits)

    def _write_parquet_files(
        self,
        df: pl.DataFrame,
        df_stats: pl.DataFrame,
        df_splits: pl.DataFrame,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        if not self.save_to.exists():
            self.save_to.mkdir(parents=True)

        (   # enforce the order provided in the schema
            df
            .select(*self.dset_schema.fields)
            .write_parquet(self.save_to / f"{self.dataset_name}.parquet")
        )
        df_stats.write_parquet(
            self.save_to / f"{self.dataset_name}_stats.parquet"
        )
        df_splits.write_parquet(
            self.save_to / f"{self.dataset_name}_splits.parquet"
        )
        return df, df_stats, df_splits


class Mirage19(Dataset):
    def __init__(self):
        super().__init__(name=DATASET_NAME.MIRAGE19)
        self.df_app_metadata = pl.read_csv(
            DATASETS_RESOURCES_FOLDER / f"{self.name}_app_metadata.csv"
        )

    @property
    def _list_raw_json_files(self):
        return list(self.folder_raw.rglob("*.json"))

#    def _parse_raw_json(
#        self, 
#        sort_by: Iterable[str],
#    ) -> pl.DataFrame:
#        files = self._list_raw_json_files
#        dset_schema = self.get_schema(DATASET_TYPE.RAW)
#
#        with tempfile.TemporaryDirectory() as tmp_folder:
#            tmp_folder = pathlib.Path(tmp_folder)
#            func = functools.partial(
#                _load_raw_json_worker, 
#                dset_schema=dset_schema, 
#                save_to=tmp_folder,
#            )
#            with (
#                richutils.Progress(
#                    description="Parse JSON files...", 
#                    total=len(files)
#                ) as progress,
#                multiprocessing.Pool(processes=2) as pool,
#            ):
#                for _ in pool.imap_unordered(func, files):
#                    progress.update()
#
#            with richutils.SpinnerProgress(description="Reload..."):
#                df = (
#                    pl.read_ndjson(
#                        tmp_folder, 
#                        schema=dset_schema.to_polars()
#                    )
#                    .sort(*sort_by)
#                )
#        return df
#
#    def raw(self) -> pl.DataFrame:
#        df = self._parse_raw_json(
#            sort_by=(
#                "parent_folder", 
#                "fname", 
#                "fname_row_idx"
#            )
#        )
#        with richutils.SpinnerProgress(description="Writing parquet files..."):
#            fileutils.save_parquet(
#                df, 
#                save_as=self.folder_raw/f"{self.name}.parquet", 
#                echo=False
#            )
#        return df
    def raw(self) -> pl.DataFrame:
        parser = BaseParserRawJSON(self.name, self.get_schema(DATASET_TYPE.RAW))
        return parser.run(
            *self._list_raw_json_files,
            sort_by=(
                "parent_folder", 
                "fname", 
                "fname_row_idx"
            ),
            save_to=self.folder_raw,
        )

#    def _raw_postprocess_rename_columns(self, df: pl.DataFrame) -> pl.DataFrame:
#        df = (
#            df
#            .rename(_rename_columns(df.columns))
#            .rename({
#                "parent_folder": "device_id"
#            })
#        )
#        return df

#    def _raw_postprocess_add_app_and_background(self, df: pl.DataFrame) -> pl.DataFrame:
#        return (
#            df
#            # add app column using static metadata
#            .join(
#                self.df_app_metadata,
#                left_on="label",
#                right_on="android_package_name",
#                how="left",
#            )
#            .with_columns(
#                # flows without a recognized label are re-labeled as background
#                app=(pl.col("app").fill_null(APP_LABEL_BACKGROUND))
#            )
#            .with_columns(
#                # force to background flows with UDP packets of size zero
#                app=(
#                    pl.when(
#                        (pl.col("proto") == "udp").and_(
#                            pl.col("pkts_size").list.min() == 0
#                        )
#                    )
#                    .then(pl.lit(APP_LABEL_BACKGROUND))
#                    .otherwise(pl.col("app"))
#                )
#            )
#        )

#    def _raw_postprocess_add_other_columns(self, df: pl.DataFrame) -> pl.DataFrame:
#        # add column: convert proto_id to string (6->tcp, 17->udp)
#        df = df.with_columns(
#            proto=(
#                pl.when(pl.col("proto_id").eq(6))
#                .then(pl.lit("tcp"))
#                .otherwise(pl.lit("udp"))
#            ),
#        )
#
#        # add columns: ip addresses private/public
#        df = curation.add_is_private_ip_columns(df)
#        # add columns: check if tcp handshake is valid
#        df = curation.add_is_valid_tcp_handshake_heuristic(
#            df, tcp_handshake_size=0, direction_upload=0, direction_download=1
#        )
#
#        return (
#            df
#            # add a global row_id
#            .with_row_index(name="row_id")
#        )


#    def _raw_postprocess(self, recompute: bool = False) -> pl.DataFrame:
#        def _get_stats(df):
#            df_stats = curation.get_stats(df)
#            return (df, df_stats)
#
#        def _write_parquet_files(tpl):
#            df, df_stats = tpl
#            df.write_parquet(
#                self.folder_raw / "_postprocess.parquet"
#            )
#            df_stats.write_parquet(
#                self.folder_raw / f"_postprocess_stats.parquet"
#            )
#            return df, df_stats
#
#        # attempt at loading the previously generate raw version
#        fname = self.folder_raw / f"{self.name}.parquet"
#        if fname.exists() and not recompute:
#            return fileutils.load_parquet(fname, echo=False)
#
#        df = self.raw()
#        # ...and triggering postprocessing steps        
#        df, _ = SequentialPipe(
#            SequentialPipeStage(
#                self._raw_postprocess_rename_columns,
#                name="Rename columns",
#            ),
#            SequentialPipeStage(
#                self._raw_postprocess_add_other_columns,
#                name="Add columns", 
#            ),
#            SequentialPipeStage(
#                self._raw_postprocess_add_app_and_background,
#                name="Add metadata",
#            ),
#            SequentialPipeStage(
#                _get_stats,
#                name="Compute statistics",
#            ),
#            SequentialPipeStage(
#                _write_parquet_files,
#                name="Write parquet files",
#            ),
#            name="Postprocess raw..."
#        ).run(df) 
#
#        return df

    def _raw_postprocess_rename_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        df = (
            df
            .rename(_rename_columns(df.columns))
            .rename({
                "parent_folder": "device_id",
                "pkts_l4_size": "pkts_size",
            })
        )
        return df

    def _raw_postprocess(self) -> pl.DataFrame:
        self.load(DATASET_TYPE.RAW)
        pipeline = BaseRawPostprocessingPipeline(
            self.df_app_metadata,
            save_to=self.folder_raw
        )
        pipeline.replace_stage(
            "Rename columns",
            SequentialPipeStage(
                self._raw_postprocess_rename_columns,
                "Rename columns",
            )
        )
        return pipeline.run(self.df)

    def _curate_rename(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.rename({
            "label": "android_package_name",
        })

#    def _curate_drop_background(self, df: pl.DataFrame) -> pl.DataFrame:
#        return df.filter(pl.col("app") != APP_LABEL_BACKGROUND)
#
#    def _curate_adjust_packet_series(self, df: pl.DataFrame) -> pl.DataFrame:
#        return df.with_columns(
#            # increase packets size to reflect the expected true size
#            # for TCP, add 40 bytes
#            # for UDP, add 28 bytes
#            pkts_size=(
#                pl.when(pl.col("proto") == "tcp")
#                .then(pl.col("pkts_size").list.eval(pl.element() + 40))
#                .otherwise(pl.col("pkts_size").list.eval(pl.element() + 28))
#            ),
#            # enforce direction (0/upload: 1, 1/download: -1)
#            pkts_dir=(
#                pl.col("pkts_dir").list.eval(
#                    pl.when(pl.element() == 0).then(1).otherwise(-1)
#                )
#            ),
#        )
#
#    def _curate_add_pkt_indices_columns(self, df: pl.DataFrame) -> pl.DataFrame:
#        return df.with_columns(
#            # series with the index of TCP acks packets
#            pkts_ack_idx=(
#                pl.when(pl.col("proto") == "tcp")
#                # for TCP, acks are enforced to 40 bytes
#                .then(curation.expr_pkts_ack_idx(ack_size=40))
#                # for UDP, packets are always larger then 0 bytes
#                # so the following is selecting all indices
#                .otherwise(curation.expr_pkts_ack_idx(ack_size=0))
#            ),
#            # series with the index of data packets
#            pkts_data_idx=(
#                pl.when(pl.col("proto") == "tcp")
#                # for TCP, acks are enforced to 40 bytes
#                .then(curation.expr_pkts_data_idx(ack_size=40))
#                # for UDP, packets are always larger then 0 bytes
#                # so the following is selecting all indices
#                .otherwise(curation.expr_pkts_data_idx(ack_size=0))
#            ),
#        )
#
#    def _curate_add_other_columns(self, df: pl.DataFrame) -> pl.DataFrame:
#        return (df
#            .with_columns(
#                # length of all series
#                pkts_len=(pl.col("pkts_size").list.len()),
#                # flag to indicate if the packet sizes have all packets
#                pkts_is_complete=(pl.col("pkts_size").list.len() == pl.col("packets")),
#                # series pkts_size * pkts_dir
#                pkts_size_times_dir=(curation.expr_pkts_size_times_dir()),
#            )
#            .with_columns(
#                # number of ack packets
#                packets_ack=(pl.col("pkts_ack_idx").list.len()),
#                # number of ack packets in upload
#                packets_ack_upload=(
#                    curation.expr_list_len_upload("pkts_size_times_dir", "pkts_ack_idx")
#                ),
#                # number of ack packets in download
#                packets_ack_download=(
#                    curation.expr_list_len_download("pkts_size_times_dir", "pkts_ack_idx")
#                ),
#                # number of data packets
#                packets_data=(pl.col("pkts_data_idx").list.len()),
#                # number of ack packets in upload
#                packets_data_upload=(
#                    curation.expr_list_len_upload("pkts_size_times_dir", "pkts_data_idx")
#                ),
#                # number of ack packets in download
#                packets_data_download=(
#                    curation.expr_list_len_download("pkts_size_times_dir", "pkts_data_idx")
#                ),
#            )
#        )
#
#    def _curate_drop_columns(self, df: pl.DataFrame) -> pl.DataFrame:
#        return df.drop(
#            [
#                "pkts_src_port",
#                "pkts_dst_port",
#                "pkts_raw_payload",
#                "labeling_type",
#            ]
#        )
#
#    def _curate_final_filter(self, df: pl.DataFrame, min_pkts: int = None) -> pl.DataFrame:
#        df_new = df.filter(
#            # flows starting with a complete handshake
#            pl.col("is_valid_handshake")
#        )
#
#        if min_pkts is not None:
#            # flows with at least a specified number of packets
#            df_new = df.filter(pl.col("packets") >= min_pkts)
#        return df_new
#
#    def curate(self) -> pl.DataFrame:
#        def _get_stats(df):
#            df_stats = curation.get_stats(df)
#            return (df, df_stats)
#
#        def _get_splits(tpl):
#            df, df_stats = tpl
#            self.df = df
#            df_splits = self.compute_splits(
#                num_splits=10,
#                test_size=0.1,
#                seed=1,
#            )
#            self.df = None
#            return (df, df_stats, df_splits)
#
#        def _write_parquet_files(tpl):
#            df, df_stats, df_splits = tpl
#            folder = self.folder_curate
#            if not folder.exists():
#                folder.mkdir(parents=True)
#            df.write_parquet(folder / f"{self.name}.parquet")
#            df_stats.write_parquet(
#                folder / f"{self.name}_stats.parquet"
#            )
#            df_splits.write_parquet(
#                folder / f"{self.name}_splits.parquet"
#            )
#            return df, df_stats, df_splits
#
#        df = self._raw_postprocess(recompute=False)
#
#        self.df, self.df_stats, self.df_splits = SequentialPipe(
#            SequentialPipeStage(
#                self._curate_rename,
#                name="Column renaming",
#            ),
#            SequentialPipeStage(
#                self._curate_drop_background, 
#                name="Drop background flows"
#            ),
#            SequentialPipeStage(
#                self._curate_adjust_packet_series,
#                name="Adjust packet series",
#            ),
#            SequentialPipeStage(
#                self._curate_add_pkt_indices_columns,
#                name="Add packet series indices"
#            ),
#            SequentialPipeStage(
#                self._curate_add_other_columns,
#                name="Add more columns",
#            ),
#            SequentialPipeStage(
#                self._curate_drop_columns,
#                name="Drop columns",
#            ),
#            SequentialPipeStage(
#                self._curate_final_filter,
#                name="Filter out flows",
#            ),
#            SequentialPipeStage(
#                _get_stats,
#                name="Compute statistics",
#            ),
#            SequentialPipeStage(
#                _get_splits,
#                name="Compute splits",
#            ),
#            SequentialPipeStage(
#                _write_parquet_files,
#                name="Write parquet files",
#            ),
#            name="Curation..."
#        ).run(df)
#
#        return self.df

    def curate(self) -> pl.DataFrame:
        fname = self.folder_raw / "_postprocess.parquet"
        if not fname.exists():
            df = self._raw_postprocess()
        else:
            with richutils.SpinnerProgress(
                description=f"Load {self.name}/raw postprocess..."
            ):
                df = fileutils.load_parquet(fname, echo=False)

        pipeline = BaseCuratePipeline(
            self.name,
            save_to=self.folder_curate,
            dset_schema=self.get_schema(DATASET_TYPE.CURATE),
        )
        self.df, self.df_stats, self.df_splits = pipeline.run(df)
        return df
        

class Mirage22(Dataset):
    def __init__(self):
        super().__init__(name=DATASET_NAME.MIRAGE22)
        self.df_app_metadata = pl.read_csv(
           DATASETS_RESOURCES_FOLDER / f"{self.name}_app_metadata.csv"
        )

    @property
    def _list_raw_json_files(self) -> List[pathlib.Path]:
        return list(
            (
                self.folder_raw 
                / "MIRAGE-COVID-CCMA-2022" 
                / "Raw_JSON"
            ).rglob("*.json")
        )

#    def install(self, no_download: bool = False) -> pathlib.Path:
#        subfolder = (
#            self.install_folder 
#            / "raw" 
#            / "MIRAGE-COVID-CCMA-2022" 
#            / "Raw_JSON"
#        )
#        #extra_unpack = (
#            subfolder / "Discord.zip",
#            subfolder / "Meet.zip",
#            subfolder / "Slack.zip",
#            subfolder / "Zoom.zip",
#            subfolder / "GotoMeeting.zip",
#            subfolder / "Messenger.zip",
#            subfolder / "Teams.zip",
#            subfolder / "Skype.zip",
#            subfolder / "Webex.zip",
#        )
#        return super().install(no_download, extra_unpack)

#    def raw(self) -> pl.DataFrame:
#        df = self._parse_raw_json(
#            sort_by=(
#                "flow_metadata_BF_device",
#                "fname", 
#                "fname_row_idx"
#            )
#        )
#        with richutils.SpinnerProgress(description="Writing parquet files..."):
#            fileutils.save_parquet(
#                df, 
#                save_as=self.folder_raw/f"{self.name}.parquet", 
#                echo=False
#            )
#        return df
    def raw(self) -> pl.DataFrame:
        parser = BaseParserRawJSON(self.name, self.get_schema(DATASET_TYPE.RAW))
        return parser.run(
            *self._list_raw_json_files,
            sort_by=(
                "flow_metadata_BF_device",
                "fname", 
                "fname_row_idx"
            ),
            save_to=self.folder_raw,
        )

    def _raw_postprocess_rename_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        df = (
            df
            .rename(_rename_columns(df.columns))
            .rename({
                "device": "device_id",
                "pkts_l3_size": "pkts_size",
                #"label": "android_package_name",
            })
        )
        return df

    def _raw_postprocess_drop_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.drop(
            "pkts_is_clear",
            "pkts_heuristic",
            "pkts_annotations",
            "label_source",
            "sublabel",
            "label_version_code",
            "label_version_name",
            "labeling_type",
            #"pkts_l4_size",
            "pkts_l4_header_size",
            "pkts_l3_header_size",
            "pkts_raw_payload",
            "pkts_src_port",
            "pkts_dst_port",
        )

#    def _raw_postprocess_clip_series(self, df: pl.DataFrame, num_packets: int = 30) -> pl.DataFrame:
#        return df.with_columns(**{
#            col: pl.col(col).list.head(num_packets)
#            for col in (
#                "pkts_timestamp",
#                "pkts_dir",
#                "pkts_size",
#                "pkts_iat",
#                "pkts_tcp_win_size",
#                "pkts_tcp_flags",
#                "pkts_l4_size",
#            )
#        })

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
        df = curation.add_is_valid_tcp_handshake_from_flags(
            df, 
            "pkts_tcp_flags",
            "pkts_dir",
            "proto", 
            proto_udp="udp",
            direction_upload=0,
            direction_download=1, 
        )

        return (
            df
            # add a global row_id
            .with_row_index(name="row_id")
        )

    def _raw_postprocess(self) -> pl.DataFrame:
        _ = self.load(DATASET_TYPE.RAW)
        df = self.df

        pipeline = BaseRawPostprocessingPipeline(
            self.df_app_metadata,
            self.folder_raw,
        )
        pipeline.replace_stage(
            "Rename columns",
            SequentialPipeStage(
                self._raw_postprocess_rename_columns,
                name="Rename columns",
            ),
        )
        pipeline.replace_stage(
            "Add columns", 
            SequentialPipeStage(
                self._raw_postprocess_add_other_columns,
                name="Add columns", 
            ),
        )
        pipeline.insert(
            1, 
            SequentialPipeStage(
                self._raw_postprocess_drop_columns,
                name="Drop columns",
            ),
        )
        return pipeline.run(df)

    def _curate_adjust_packet_series(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            # enforce direction (0/upload: 1, 1/download: -1)
            pkts_dir=(
                pl.col("pkts_dir").list.eval(
                    pl.when(pl.element() == 0).then(1).otherwise(-1)
                )
            ),
        )

    def _curate_add_pkt_indices_columns(self, df: pl.DataFrame) -> pl.DataFrame:
#        return df.with_columns(
#            # series with the index of TCP acks packets
#            pkts_ack_idx=(
#                pl.when(pl.col("proto") == "tcp")
#                # for TCP, acks are enforced to 40 bytes
#                .then(curation.expr_pkts_ack_idx("pkts_l4_size", ack_size=0))
#                # for UDP, packets are always larger then 0 bytes
#                # so the following is selecting all indices
#                .otherwise(curation.expr_pkts_ack_idx("pkts_l4_size", ack_size=-1))
#            ),
#            # series with the index of data packets
#            pkts_data_idx=(
#                pl.when(pl.col("proto") == "tcp")
#                # for TCP, acks are enforced to 40 bytes
#                .then(curation.expr_pkts_data_idx("pkts_l4_size", ack_size=0))
#                # for UDP, packets are always larger then 0 bytes
#                # so the following is selecting all indices
#                .otherwise(curation.expr_pkts_data_idx("pkts_l4_size", ack_size=-1))
#            ),
#        )
        return df.with_columns(
            # series with the index of TCP acks packets
            pkts_ack_idx=(
                pl.when(pl.col("proto") == "tcp")
                # for TCP, acks are enforced to 40 bytes
                .then(
                    curation.expr_indices_list_value_equal_to(
                        "pkts_l4_size", 
                        value=0
                    )
                )
                # for UDP, there are not ACK
                .otherwise(
                    curation.expr_indices_list_value_lower_than(
                        "pkts_l4_size", 
                        value=0, 
                        inclusive=False
                    )
                )
            ),
            # series with the index of data packets
            pkts_data_idx=(
                pl.when(pl.col("proto") == "tcp")
                # for TCP, acks are enforced to 40 bytes
                .then(
                    curation.expr_indices_list_value_not_equal_to(
                        "pkts_l4_size", 
                        value=0
                    )
                )
                # for UDP, all packets are data packets
                .otherwise(
                    curation.expr_indices_list_value_greater_than(
                        "pkts_l4_size", 
                        value=0, 
                        inclusive=True
                    )
                )
            )
        )

    def _curate_drop_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.drop(
            "pkts_l4_size",
            "proto_id",
        )


#    def curate(self) -> pl.DataFrame:
#        def _get_stats(df):
#            df_stats = curation.get_stats(df)
#            return (df, df_stats)
#
#        def _get_splits(tpl):
#            df, df_stats = tpl
#            self.df = df
#            df_splits = self.compute_splits(
#                num_splits=10,
#                test_size=0.1,
#                seed=1,
#            )
#            self.df = None
#            return (df, df_stats, df_splits)
#
#        def _write_parquet_files(tpl):
#            df, df_stats, df_splits = tpl
#            folder = self.folder_curate
#            if not folder.exists():
#                folder.mkdir(parents=True)
#            df.write_parquet(folder / f"{self.name}.parquet")
#            df_stats.write_parquet(
#                folder / f"{self.name}_stats.parquet"
#            )
#            df_splits.write_parquet(
#                folder / f"{self.name}_splits.parquet"
#            )
#            return df, df_stats, df_splits
#
#        df = self._raw_postprocess() 
#
#        self.df, self.df_stats, self.df_splits = SequentialPipe(
#            #SequentialPipeStage(
#            #    self._curate_rename,
#            #    name="Column renaming",
#            #),
#            SequentialPipeStage(
#                self._curate_drop_background, 
#                name="Drop background flows"
#            ),
#            SequentialPipeStage(
#                self._curate_adjust_packet_series,
#                name="Adjust packet series",
#            ),
#            SequentialPipeStage(
#                self._curate_add_pkt_indices_columns,
#                name="Add packet series indices"
#            ),
#            SequentialPipeStage(
#                self._curate_add_other_columns,
#                name="Add more columns",
#            ),
#            SequentialPipeStage(
#                self._curate_drop_columns,
#                name="Drop columns",
#            ),
#            SequentialPipeStage(
#                self._curate_final_filter,
#                name="Filter out flows",
#            ),
#            SequentialPipeStage(
#                _get_stats,
#                name="Compute statistics",
#            ),
#            SequentialPipeStage(
#                _get_splits,
#                name="Compute splits",
#            ),
#            SequentialPipeStage(
#                _write_parquet_files,
#                name="Write parquet files",
#            ),
#            name="Curation..."
#        ).run(df)
#
#        return self.df


    def curate(self) -> pl.DataFrame:
        fname = self.folder_raw / "_postprocess.parquet"
        if not fname.exists():
            df = self._raw_postprocess()
        else:
            with richutils.SpinnerProgress(
                description=f"Load {self.name}/raw postprocess..."
            ):
                df = fileutils.load_parquet(fname, echo=False)

        pipeline = BaseCuratePipeline(
            self.name,
            save_to=self.folder_curate,
            dset_schema=self.get_schema(DATASET_TYPE.CURATE),
        )
        pipeline.replace_stage(
            "Adjust packet series",
            SequentialPipelineStage(
                self._curate_adjust_packet_series,
                name="Adjust packet series",
            ),
        )
        pipeline.replace_stage(
            "Add packet series indices",
            SequentialPipelineStage(
                self._curate_add_pkt_indices_columns,
                name="Add packet series indices"
            ),
        )
        pipeline.replace_stage(
            "Drop columns",
            SequentialPipelineStage(
                self._curate_drop_columns,
                name="Drop columns",
            )
        )

        self.df, self.df_stats, self.df_splits = pipeline.run(df)
        return self.df
            

        #def _curate_adjust_packet_series(self, df: pl.DataFrame) -> pl.DataFrame:
        #def _curate_add_pkt_indices_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        #def _curate_drop_columns(self, df: pl.DataFrame) -> pl.DataFrame:

#            SequentialPipelineStage(
#                self._drop_background, 
#                name="Drop background flows"
#            ),
#            SequentialPipelineStage(
#                self._add_other_columns,
#                name="Add more columns",
#            ),
#            SequentialPipelineStage(
#                self._final_filter,
#                name="Filter out flows",
#            ),
#            SequentialPipelineStage(
#                self._compute_stats,
#                name="Compute statistics",
#            ),
#            SequentialPipelineStage(
#                self._compute_splits,
#                name="Compute splits",
#            ),
#            SequentialPipelineStage(
#                self._write_parquet_files,
#                name="Write parquet files",
#            ),
#        ]





#            #SequentialPipeStage(
#            #    self._curate_rename,
#            #    name="Column renaming",
#            #),
#            SequentialPipeStage(
#                self._curate_drop_background, 
#                name="Drop background flows"
#            ),
#            SequentialPipeStage(
#                self._curate_adjust_packet_series,
#                name="Adjust packet series",
#            ),
#            SequentialPipeStage(
#                self._curate_add_pkt_indices_columns,
#                name="Add packet series indices"
#            ),
#            SequentialPipeStage(
#                self._curate_add_other_columns,
#                name="Add more columns",
#            ),
#            SequentialPipeStage(
#                self._curate_drop_columns,
#                name="Drop columns",
#            ),
#            SequentialPipeStage(
#                self._curate_final_filter,
#                name="Filter out flows",
#            ),
#            SequentialPipeStage(
#                _get_stats,
#                name="Compute statistics",
#            ),
#            SequentialPipeStage(
#                _get_splits,
#                name="Compute splits",
#            ),
#            SequentialPipeStage(
#                _write_parquet_files,
#                name="Write parquet files",
#            ),
#            name="Curation..."





