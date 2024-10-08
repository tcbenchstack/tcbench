from __future__ import annotations

import polars as pl

import pathlib
import multiprocessing
import functools

from tcbench.libtcdatasets.core import (
    Dataset,
    DatasetSchema
)
from tcbench.libtcdatasets.constants import DATASET_NAME, DATASET_TYPE
from tcbench.cli import richutils


def load_raw_txt(
    path: pathlib.Path, 
) -> pl.DataFrame:
    import tcbench
    dset_schema = (
        tcbench.datasets_catalog()
        [DATASET_NAME.UCDAVIS19]
        .get_schema(DATASET_TYPE.RAW)
    )
    return pl.read_csv(path, separator="\t", schema=dset_schema.to_polars())


def _parse_raw_txt_worker(
    path: pathlib.Path, 
    schema: pl.Schema
) -> pl.DataFrame:
    df = pl.read_csv(path, separator="\t", schema=schema)
    df2 = pl.DataFrame({
        "unixtime": [df["unixtime"].to_list()],
        "pkts_timetofirst": [df["timetofirst"].to_list()],
        "pkts_size": [df["packet_size"].to_list()],
        "pkts_dir": [df["packet_dir"].to_list()],
        "app": path.parent.name.lower().replace(" ", "_"),
        "fname": path.name,
        "partition": (
            path
            .parent
            .parent
            .name
            .lower()
            .replace(")", "")
            .replace("(", "-")
        )
    })
    return df2


class UCDavis19(Dataset):
    def __init__(self):
        super().__init__(name=DATASET_NAME.UCDAVIS19)

    @property
    def _list_raw_txt_files(self):
        return list(self.folder_raw.rglob("*.txt"))

    def raw(self):
        files = self._list_raw_txt_files
        with (
            richutils.Progress(description="Parse raw...", total=len(files)) as progress,
            multiprocessing.Pool(processes=2) as pool,
        ):
            schema = self.get_schema(DATASET_TYPE.RAW).to_polars()
            func = functools.partial(_parse_raw_txt_worker, schema=schema)
            data = []
            for df in pool.imap_unordered(func, files):
                data.append(df)
                progress.update()

        with richutils.SpinnerProgress(description="Writing parquet files..."):
            df = pl.concat(data).with_row_index("row_id")
            df.write_parquet(self.folder_raw / f"{self.name}.parquet")
            
    def curate(self):
        pass
