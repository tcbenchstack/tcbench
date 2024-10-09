from __future__ import annotations

import polars as pl

import pathlib
import multiprocessing
import functools

from tcbench import fileutils
from tcbench.cli import richutils
from tcbench.libtcdatasets.core import (
    Dataset,
    DatasetSchema,
    BaseDatasetProcessingPipeline,
    SequentialPipelineStage,
)
from tcbench.libtcdatasets.constants import DATASET_NAME, DATASET_TYPE

PARTITION_PRETRAINING = "pretraining"
PARTITION_SCRIPT = "retraining-script-triggered"
PARTITION_HUMAN = "retraining-human-triggered"


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
    df2 = (
        pl.DataFrame({
            "pkts_timestamp": [df["unixtime"].to_list()],
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
    )
    return df2

class RawPostorocessingPipeline(BaseDatasetProcessingPipeline):
    def __init__(self, save_to: pathlib.Path):
        super().__init__(
            dataset_name=DATASET_NAME.UCDAVIS19,
            description="Postprocess raw...",
            save_to=save_to,
        )

        stages = [
            SequentialPipelineStage(
                self._add_columns,
                "Adding columns",
            ),
            SequentialPipelineStage(
                self._compute_stats,
                "Computing stats",
            ),
            SequentialPipelineStage(
                functools.partial(
                    self._write_parquet_files,
                ),
                "Writing parquet files",
            )
        ]
        self.clear()
        self.extend(stages)

    def _add_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return (
            df
            .with_columns(
                bytes=pl.col("pkts_size").list.sum(),
                packets=pl.col("pkts_size").list.len(),
                duration=(
                    pl.col("pkts_timestamp").list.last() 
                    - pl.col("pkts_timestamp").list.first()
                ),
                proto=pl.lit("udp"),
                is_valid_handshake=pl.lit(True),
            )
            .with_row_index("row_id")
        )

class CuratePipeline(BaseDatasetProcessingPipeline):
    def __init__(self, save_to: pathlib.Path):
        super().__init__(
            dataset_name=DATASET_NAME.UCDAVIS19,
            description="Curate...",
            save_to=save_to,
        )

    def run(self, df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
        df_pretrain = df.filter(pl.col("partition") == PARTITION_PRETRAINING)
        df_human = df.filter(pl.col("partition") == PARTITION_HUMAN)
        df_script = df.filter(pl.col("partition") == PARTITION_SCRIPT)
        res = dict()

        self.clear()
        self.name = "Curate pretrain partition..."
        self.extend((
            SequentialPipelineStage(
                self._compute_stats,
                "Compute stats",
            ),
            SequentialPipelineStage(
                self._compute_splits,
                "Compute splits",
            ),
            SequentialPipelineStage(
                functools.partial(
                    self._write_parquet_files,
                    fname_prefix=self.dataset_name
                ),
                "Write parquet files",
            ),
        ))
        _, df_stats, df_splits = super().run(df_pretrain)
        res["df"] = df_pretrain
        res["df_stats"] = df_stats
        res["df_splits"] = df_splits

        for df_curr, suffix in zip(
            (df_human, df_script), ("human", "script")
        ):
            self.clear()
            self.name = f"Curate {suffix} partition..."
            self.extend((
                SequentialPipelineStage(
                    self._compute_stats,
                    "Compute stats",
                ),
                SequentialPipelineStage(
                    functools.partial(
                        self._write_parquet_files,
                        fname_prefix=f"{self.dataset_name}_{suffix}"
                    ),
                    "Write parquet files",
                ),
            ))
            _1, df_stats, _2 = super().run(df_curr)
            res[f"df_{suffix}"] = df_curr
            res[f"df_{suffix}_stats"] = df_stats
        
        return res


class UCDavis19(Dataset):
    def __init__(self):
        super().__init__(name=DATASET_NAME.UCDAVIS19)
        self.df = None
        self.df_stats = None
        self.df_splits = None
        self.df_human = None
        self.df_human_stats = None
        self.df_script = None
        self.df_script_stats = None

    @property
    def _list_raw_txt_files(self):
        return list(self.folder_raw.rglob("*.txt"))

    def _parse_raw(self) -> pd.DataFrame:
        paths = self._list_raw_txt_files
        with (
            richutils.Progress(
                description="Parsing raw txt files...", 
                total=len(paths),
            ) as progress,
            multiprocessing.Pool(processes=2) as pool,
        ):
            schema = self.get_schema(DATASET_TYPE.RAW).to_polars()
            func = functools.partial(_parse_raw_txt_worker, schema=schema)
            data = []
            for df in pool.imap_unordered(func, paths):
                data.append(df)
                progress.update()
            return pl.concat(data)

    def raw(self) -> pl.DataFrame:
        self.df = None
        self.df_stats = None
        self.df_splits = None
        self.df_human = None
        self.df_human_stats = None
        self.df_script = None
        self.df_script_stats = None

        self.df = self._parse_raw()
        with richutils.SpinnerProgress(description="Writing parquet files"):
            fileutils.save_parquet(
                self.df, 
                self.folder_raw / f"{self.name}.parquet", 
                echo=False
            )
        return self.df
        self.df, self.df_stats = (
            RawPostorocessingPipeline(
                save_to=self.folder_raw
            )
            .run(df)
        )
        return self.df
  
    def _raw_postprocess(self) -> pl.DataFrame:
        self.load(DATASET_TYPE.RAW)
        df, _ = RawPostorocessingPipeline(
            save_to=self.folder_raw
        ).run(self.df)
        return df
        
    def curate(self) -> pl.DataFrame:
        fname = self.folder_raw / f"_postprocess.parquet"
        if not fname.exists():
            df = self._raw_postprocess()
        else:
            with richutils.SpinnerProgress(
                description=f"Load {self.name}/raw postprocess..."
            ):
                df = fileutils.load_parquet(fname, echo=False)

        schema = self.get_schema(DATASET_TYPE.CURATE).to_polars()
        df = df.select(list(schema.keys()))

        res = CuratePipeline(save_to=self.folder_curate).run(df)
        self.df = res["df"]
        self.df_stats = res["df_stats"]
        self.df_splits = res["df_splits"]
        self.df_human = res["df_human"]
        self.df_human_stats = res["df_human_stats"]
        self.df_script = res["df_script"]
        self.df_script_stats = res["df_script_stats"]
        return self.df

    def load(self, dset_type: DATASET_TYPE, *args, **kwargs) -> Dataset:
        self.df = None
        self.df_stats = None
        self.df_splits = None
        self.df_human = None
        self.df_human_stats = None
        self.df_script = None
        self.df_script_stats = None

        super().load(dset_type, *args, **kwargs)

        if dset_type == DATASET_TYPE.CURATE:
            self.df_human = fileutils.load_if_exists(
                self.folder_curate / f"{self.name}_human.parquet",
                echo=False
            )
            self.df_human_stats = fileutils.load_if_exists(
                self.folder_curate / f"{self.name}_human_stats.parquet",
                echo=False
            )
            self.df_script = fileutils.load_if_exists(
                self.folder_curate / f"{self.name}_script.parquet",
                echo=False
            )
            self.df_script_stats = fileutils.load_if_exists(
                self.folder_curate / f"{self.name}_script_stats.parquet",
                echo=False
            )
        return self
