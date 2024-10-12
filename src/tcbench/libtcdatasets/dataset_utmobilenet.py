from __future__ import annotations

from typing import Tuple
from collections import OrderedDict

import polars as pl
import numpy as np

import pathlib
import multiprocessing
import functools
import shutil
import tempfile
import csv

from tcbench import fileutils
from tcbench.cli import richutils
from tcbench.libtcdatasets.core import (
    Dataset,
    DatasetSchema,
    BaseDatasetProcessingPipeline,
    SequentialPipelineStage,
    RawDatasetInstaller,
    _remove_fields_from_schema,
)
from tcbench.libtcdatasets.constants import DATASET_NAME, DATASET_TYPE
from tcbench.libtcdatasets import curation


def _fix_raw_csv(path: pathlib.Path) -> Iterable[Dict[str, str]]:
    with open(path) as fin:
        lines = []
        for row in csv.DictReader(fin):
            if "," not in row["ip.hdr_len"]:
                lines.append(row)
                continue
            row1 = row.copy()
            row2 = row.copy()
            for field in (
                "ip.hdr_len",
                "ip.dsfield.ecn",
                "ip.len",
                "ip.id",
                "ip.frag_offset",
                "ip.ttl",
                "ip.proto",
                "ip.checksum",
                "ip.src",
                "ip.dst",
                "tcp.hdr_len",
                "tcp.len",
                "tcp.srcport",
                "tcp.dstport",
                "tcp.seq",
                "tcp.ack",
                "tcp.flags.ns",
                "tcp.flags.fin",
                "tcp.window_size_value",
                "tcp.checksum",
                "tcp.urgent_pointer",
                "tcp.option_kind",
                "tcp.option_len",
                "tcp.options.timestamp.tsval",
                "tcp.options.timestamp.tsecr",
                "udp.srcport",
                "udp.dstport",
                "udp.length",
                "udp.checksum",
                "gquic.puflags.rsv",
                "gquic.packet_number",
            ):
                value = row[field]
                parts = value.split(",")
                if len(parts) == 1:
                    continue
                row1[field] = parts[0]
                row2[field] = parts[1]
            lines.append(row1)
            lines.append(row2)

        # wiping out L4 information for ICMP records
        for row in lines:
            if row["ip.proto"] != 1:
                continue
            for field in (
                "tcp.hdr_len"
                "tcp.len"
                "tcp.srcport"
                "tcp.dstport"
                "tcp.seq"
                "tcp.ack"
                "tcp.flags.ns"
                "tcp.flags.fin"
                "tcp.window_size_value"
                "tcp.checksum"
                "tcp.urgent_pointer"
                "tcp.option_kind"
                "tcp.option_len"
                "tcp.options.timestamp.tsval"
                "tcp.options.timestamp.tsecr"
                "udp.srcport"
                "udp.dstport"
                "udp.length"
                "udp.checksum"
                "gquic.puflags.rsv"
                "gquic.packet_number"
            ):
                row[field] = ""
    return lines
    

def _parse_raw_csv_worker(
    path: pathlib.Path, 
    schema: pl.Schema,
    save_to: pathlib.Path = None
) -> pl.DataFrame:
    path = pathlib.Path(path)
    schema = _remove_fields_from_schema(
        schema,
        # the csv does not have these two fields   
        "fname",
        "folder"
    )
    raw_schema = OrderedDict()
    for field_name, field_type in schema.items():
        # the first column of the CSV does not have a name
        if field_name == "index":
            field_name = ""
        raw_schema[field_name] = field_type

    rows = _fix_raw_csv(path)

    with tempfile.NamedTemporaryFile("w") as tmp_file:
        fields = list(raw_schema.keys())
        tmp_file.write(",".join(fields))
        writer = csv.DictWriter(
            tmp_file, 
            fieldnames=fields,
        )
        for row in rows:
            writer.writerow(row)    
        tmp_file.flush()

        df = (
            pl
            .read_csv(tmp_file.name, schema=raw_schema)
            .rename({"": "index"})
            .with_columns(
                folder=pl.lit(str(path.parent.name)),
                fname=pl.lit(str(path.name)),
            )
        )

    if save_to is not None:
        df.write_parquet(
            save_to 
            / f"""str(path.parent.name).lower().replace(" ", "_")_{path.name}"""
        )
    return df


def load_raw_csv(path: pathlib.Path) -> pl.DataFrame:
    import tcbench
    path = pathlib.Path(path)
    schema = tcbench.get_dataset_polars_schema(
        DATASET_NAME.UTMOBILENET21,
        DATASET_TYPE.RAW,
    )
    return _parse_raw_csv_worker(path, schema)



class RawCSVParser:
    def __init__(self):
        import tcbench
        self.name = DATASET_NAME.UTMOBILENET21
        self.dset_schema = (
            tcbench.datasets_catalog()
            [DATASET_NAME.UTMOBILENET21]
            .get_schema(DATASET_TYPE.RAW)
        )

    def _parse_raw_csv(
        self, 
        *files: Iterable[pathlib.Path],
    ) -> pl.DataFrame:
        with tempfile.TemporaryDirectory() as tmp_folder:
            tmp_folder = pathlib.Path(tmp_folder)
            func = functools.partial(
                _parse_raw_csv_worker, 
                schema=self.dset_schema.to_polars(), 
                save_to=tmp_folder,
            )
            with (
                richutils.Progress(
                    description="Parse CSV files...", 
                    total=len(files)
                ) as progress,
                multiprocessing.Pool(processes=2) as pool,
            ):
                for _ in pool.imap_unordered(func, files):
                    progress.update()
            
            with richutils.SpinnerProgress(description="Reload..."):
                df = (
                    pl
                    .read_parquet(tmp_folder)
                    .sort(
                        "folder",
                        "fname",
                        "frame.number",
                    )
                )
        return df

    def run(
        self, 
        *files: Iterable[pathlib.Path], 
        save_to: pathlib.Path = None,
    ) -> pl.DataFrame:
        df = self._parse_raw_csv(*files)
        if save_to is None:
            save_to = pathlib.Path(".")
        with richutils.SpinnerProgress(description="Writing parquet files..."):
            fileutils.save_parquet(
                df, 
                save_as=save_to/f"{self.name}.parquet", 
                echo=False
            )
        return df


class UTMobilenet21(Dataset):
    def __init__(self):
        super().__init__(name=DATASET_NAME.UTMOBILENET21)

    def _install_raw(
        self, 
    ) -> pathlib.Path:
        url = self.metadata.raw_data_url_hidden[0]["UTMobileNet2021.zip"]
        UTMobilenet21DatasetInstaller(
            url=url,
            install_folder=self.install_folder,
            verify_tls=True,
            force_reinstall=True,
        )
        return self.install_folder

    @property
    def _list_raw_csv_files(self) -> Tuple[pathlib.Path]:
        return tuple(self.folder_raw.rglob("*.csv"))

    def raw(self) -> pl.DataFrame:
        return RawCSVParser().run(
            *self._list_raw_csv_files,
            save_to=self.folder_raw
        )
