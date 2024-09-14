from __future__ import annotations

import polars as pl

import pathlib

from typing import Tuple, List, Dict, Any, Iterable, Callable
from numpy.typing import NDArray
from sklearn.preprocessing import LabelEncoder

from dataclasses import dataclass

from tcbench import fileutils
from tcbench.modeling import (
    splitting, 
    datafeatures,
)
from tcbench.libtcdatasets.core import (
    Dataset,
)
from tcbench.modeling.datafeatures import (
    DEFAULT_EXTRA_COLUMNS
)
from tcbench.modeling.columns import (
    COL_APP,
    COL_ROW_ID,
)
from tcbench.modeling.constants import (
    MLMODEL_NAME,
)


class MLDataLoader:
    def __init__(
        self,
        dset: Dataset,
        features: Iterable[str],
        df_splits: pl.DataFrame,
        split_index: int = 1,
        y_colname: str = COL_APP,
        index_colname: str = COL_ROW_ID,
        series_len: int = None,
        series_pad: int = None,
        extra_colnames: Iterable[str] = DEFAULT_EXTRA_COLUMNS,
        shuffle_train: bool = True,
        seed: int = 1
    ):
        self.dset = dset
        self.y_colname = y_colname
        self.index_colname = index_colname
        self._labels = dset.df[y_colname].unique().sort().to_list()
        self.df_splits = df_splits
        self.split_index = split_index
        self.features = features
        self.extra_colnames = extra_colnames
        self.series_len = series_len
        self.series_pad = series_pad
        self.shuffle_train = shuffle_train
        self.seed = seed

        self._df_train, self._df_test = splitting.get_train_test_splits(
            self.dset.df,
            self.df_splits,
            self.split_index,
            self.index_colname
        )

        self._X_train, self._y_train, self._df_train_feat = \
            self.dataprep(
                self._df_train,
                shuffle=shuffle_train,
                seed=seed
            )
        self._X_test, self._y_test, self._df_test_feat = \
            self.dataprep(
                self._df_test,
                shuffle=False,
            )

        self._feature_names = [
            col
            for col in self._df_train_feat.columns
            if col not in (self.y_colname, *self.extra_colnames)
        ]

    @property
    def labels(self) -> List[str]:
        return self._labels

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    @property
    def train(self) -> pl.DataFrame:
        return self._df_train

    @property
    def train_feat(self) -> pl.DataFrame:
        return self._df_train_feat

    @property
    def X_train(self) -> NDArray:
        return self._X_train

    @property
    def y_train(self) -> NDArray:
        return self._y_train

    @property
    def test(self) -> pl.DataFrame:
        return self._df_test

    @property
    def X_test(self) -> NDArray:
        return self._X_test

    @property
    def y_test(self) -> NDArray:
        return self._y_test
        
    @property
    def test_feat(self) -> pl.DataFrame:
        return self._df_test_feat

    def dataprep(
        self, 
        df: pl.DataFrame, 
        shuffle: bool = False, 
        seed: int = 1
    ) -> Tuple[NDArray, NDArray, pl.DataFrame]:
        if shuffle:
            df = df.sample(
                fraction=1, 
                shuffle=True, 
                seed=seed,
            )
        return datafeatures.features_dataprep(
                df,
                self.features,
                self.series_len,
                self.series_pad,
                self.y_colname,
                self.extra_colnames,
            )


class ClassificationResults:
    def __init__(
        self, 
        model: MLModel,
        df_feat: pl.DataFrame,
        y_true: NDArray,
        y_pred: NDArray,
        split_index: int = None,
        name: str = "test",
    ):
        self.model = model
        self.name = name
        self.df_feat = df_feat.with_columns(
            y_true=pl.Series(y_true),
            y_pred=pl.Series(y_pred),
            split_index=pl.Series(split_index) if split_index else None
        ) 

    @property
    def labels(self) -> List[str]:
        return self.model.labels

    @property
    def y_true(self) -> NDArray:
        return self.df_feat["y_true"].to_numpy()

    @property
    def y_pred(self) -> NDArray:
        return self.df_feat["y_pred"].to_numpy()

    def save(self, save_to: pathlib.Path, name: str = "test") -> ClassificationResults:
        return self

    @classmethod
    def load(cls, folder: pathlib.Path, name: str = "test") -> ClassificationResults:
        return None


class MLModel:
    def __init__(
        self,
        labels: Iterable[str],
        feature_names: Iterable[str],
        model_class: Callable,
        seed: int = 1,
        **hyperparams: Dict[str, Any],
    ):
        self.labels = labels
        self.feature_names = feature_names
        self.hyperparams = hyperparams
        self.seed = seed

        self._label_encoder = self._fit_label_encoder(self.labels)
        self._model = model_class(**hyperparams)

    @property
    def name(self) -> str:
        self._model.__class__.__name__

    def _fit_label_encoder(self, labels) -> LabelEncoder:
        label_encoder = LabelEncoder()
        label_encoder.fit(self.labels)
        return label_encoder

    def encode_y(self, y: str | Iterable[str]) -> NDArray:
        if isinstance(y, str):
            y = [y]
        return self._label_encoder.transform(y)

    def decode_y(self, y: int | Iterable[int]) -> NDArray:
        if isinstance(y, int):
            y = [y]
        return self._label_encoder.inverse_transform(y)

    def fit(self, X: NDArray, y: NDArray) -> NDArray:
        self._model.fit(X, self.encode_y(y))
        return self.decode_y(self._model.predict(X))

    def predict(self, X) -> ClassificationResults:
        return self.decode_y(self._model.predict(X))

    @classmethod
    def load(cls, path: pathlib.Path) -> MLModel:
        path = pathlib.Path(path)
        if path.is_dir():
            path /= "tcbench_model.pkl"
        return fileutils.load_pickle(path)

    def save(self, save_to: pathlib.Path) -> MLModel:
        fileutils.save_pickle(self, save_to)
        return self

    @property
    def size(self) -> int:
        return None


class MLTrainer:
    def __init__(
        self,
        model: MLModel,
        dataloader: MLDataLoader,
        split_index: int = None
    ):
        self.model = model
        self.dataloader = dataloader
        self.split_index = split_index

    def fit(self, name: str = "train") -> ClassificationResults:
        y_pred = self.model.fit(
            self.dataloader.X_train,
            self.dataloader.y_train
        )
        return ClassificationResults(
            self.model,
            name=name,
            df_feat=self.dataloader.train_feat,
            y_true=self.dataloader.y_train,
            y_pred=y_pred,
            split_index=self.split_index,
        )

    def predict(self, name: str = "test") -> ClassificationResults:
        y_pred = self.model.predict(
            self.dataloader.X_test,
        )
        return ClassificationResults(
            self.model,
            name=name,
            df_feat=self.dataloader.test_feat,
            y_true=self.dataloader.y_test,
            y_pred=y_pred,
            split_index=self.split_index,
        )


