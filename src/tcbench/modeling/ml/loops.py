from __future__ import annotations

import polars as pl

from typing import Iterable

from tcbench.libtcdatasets import (
    DATASET_NAME,
    DATASET_TYPE,
    datasets_catalog,
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


def train_loop(
    dset: Dataset,
    df_splits: pl.DataFrame,
    method_name: MODELING_METHOD_NAME,
    features: Iterable[str],
    series_len: int = 10,
    seed: int = 1,
    save_to: pathlib.Path = None,
    track_train: bool = False,
) -> MultiClassificationResults:
    ldr = MLDataLoader(
        dset,
        features=features,
        df_splits=df_splits,
        split_index=1,
        series_len=series_len,
        series_pad=None,
        extra_colnames=(COL_BYTES, COL_PACKETS, COL_ROW_ID),
        shuffle_train=True,
        seed=seed
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
