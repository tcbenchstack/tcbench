from __future__ import annotations

import polars as pl
import itertools

from typing import Iterable

from tcbench import (
    DATASET_NAME,
    DATASET_TYPE,
    get_dataset,
)
from tcbench.libtcdatasets.core import Dataset
from tcbench.modeling import (
    MODELING_METHOD_NAME, 
    mlmodel_factory,
)
from tcbench.modeling.columns import (
    COL_BYTES,
    COL_PACKETS,
    COL_ROW_ID,
)
from tcbench.modeling.ml.core import (
    MLDataLoader,
    MLTrainer,
    ClassificationResults,
    MultiClassificationResults,
)

DEFAULT_TRACK_EXTRA_COLUMNS = (
    COL_BYTES, 
    COL_PACKETS, 
    COL_ROW_ID
)

def _train_loop_worker(
    dataset_name: DATASET_NAME,
    dataset_type: DATASET_TYPE,
    method_name: MODELING_METHOD_NAME,
    features: Iterable[str],
    series_len: int = 10,
    series_pad: int = None,
    seed: int = 1,
    save_to: pathlib.Path = None,
    split_index: int = 1,
    track_train: bool = False,
    track_extra_columns: Iterable[str] = DEFAULT_TRACK_EXTRA_COLUMNS,
) -> MultiClassificationResults:
    if track_extra_columns is None:
        track_extra_columns = []

    dset = get_dataset(dataset_name)

    columns = [dset.y_colname, dset.index_colname]
    for col in itertools.chain(features, track_extra_columns):
        col = str(col)
        if col not in columns:
            columns.append(col)

    dset.load(
        dataset_type, 
        min_packets=series_len,
        columns=columns
    )

    ldr = MLDataLoader(
        dset,
        features=features,
        df_splits=dset.df_splits,
        split_index=split_index,
        series_len=series_len,
        series_pad=series_pad,
        extra_colnames=track_extra_columns,
        shuffle_train=True,
        seed=seed,
    )

    mdl = mlmodel_factory(
        method_name,
        labels=ldr.labels,
        feature_names=ldr.feature_names,
        seed=seed,
    )
    trainer = MLTrainer(mdl, ldr)

    clsres_train = trainer.fit(
        save_to=save_to if track_train else None,
        name="train"
    )
    clsres_test = trainer.predict(save_to=save_to, name="test")
    mdl.save(save_to)

    clsres = MultiClassificationResults(
        train=clsres_train,
        test=clsres_test,
        model=mdl,
    )

    return clsres


def train_loop(
    dataset_name: DATASET_NAME,
    dataset_type: DATASET_TYPE,
    method_name: MODELING_METHOD_NAME,
    features: Iterable[str],
    series_len: int = 10,
    seed: int = 1,
    save_to: pathlib.Path = None,
    track_train: bool = False,
    num_workers: int = 1,
) -> MultiClassificationResults:
    return _train_loop_worker(
        dataset_name=dataset_name,
        dataset_type=dataset_type,
        method_name=method_name,
        features=features,
        series_len=series_len,
        seed=seed,
        save_to=save_to,
        track_train=track_train
    )
