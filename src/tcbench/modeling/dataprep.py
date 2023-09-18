"""
This modules contains the a hierarchy of classes
for composing Datasets and a variety of 
function to load those objects from file

All datasets are inherited from an archetype
class named FlowpicDataset.
This is a wrapper around a pandas DataFrame
and offer functionality to create flowpic
representation based on raw time series.

Two other classes are then created
to apply transformation functions

Specifically:
- AugmentWhenLoadingDataset: this class applies 
    transformations when instanciated.
    This is useful when performing supervised training

- MultiViewDataset: this class applies
    multi-view transformation on-the-fly.
    This is useful when performing contrastive learning training
"""
from __future__ import annotations

import torchvision.transforms as T
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from typing import Tuple, Dict, Callable, Any
from collections import OrderedDict
from copy import deepcopy
from enum import Enum

import abc
import torch
import torchvision
import pathlib
import functools
import multiprocessing
import logging
import sys
import logging

import tcbench
from tcbench.modeling import MODELING_DATASET_TYPE, MODELING_INPUT_REPR_TYPE
from tcbench.modeling import augmentation
from tcbench.modeling import utils
from tcbench import cli
from tcbench.libtcdatasets import datasets_utils

##class DatasetType(Enum):
# class MODELING_DATASET_TYPE(Enum):
#    """An enumeration to specify which type of dataset to load"""
#    TRAIN_VAL = "train_val_datasets"
#    TEST = "test_dataset"
#    TRAIN_VAL_LEFTOVER = "train_val_leftover_dataset"
#    FINETUNING = "for_finetuning_dataset"
#
# class INPUT_REPRESENTATION_TYPE(Enum):
#    FLOWPIC = "flowpic"
#    PKTS_TIME_SERIES = "pktseries"


#def _create_df_to_normalize_pkt_series(
#    df: pd.DataFrame,
#    timetofirst_colname: str,
#    pkts_size_colname: str,
#    pkts_dir_colname: str,
#    max_n_pkts=10,
#):
#    max_n_pkts = int(max_n_pkts)
#    series_iat = df[timetofirst_colname].apply(
#        lambda l: [0] + [j - i for i, j in zip(l[:-1], l[1:])]
#    )
#    df_pkt_size = pd.DataFrame(df[pkts_size_colname].str[:max_n_pkts].tolist())
#    df_iat = pd.DataFrame(series_iat.str[:max_n_pkts].tolist())
#    # df_iat = pd.DataFrame(series_iat.to_list())
#    df_pkt_dir = pd.DataFrame(df[pkts_dir_colname].str[:max_n_pkts].tolist())
#    df_normalize = df_pkt_size.merge(
#        df_iat, left_index=True, right_index=True, suffixes=("_size", "_iat")
#    )
#    df_normalize = df_normalize.merge(
#        df_pkt_dir, left_index=True, right_index=True, suffixes=("", "_dir")
#    )
#    df_normalize.columns = [str(i) for i in range(len(df_normalize.columns))]
#    return df_normalize

def _clip_and_pad(arr, n_pkts, pad_value=0):
    if len(arr) >= n_pkts:
        return arr[:n_pkts]
    return np.concatenate((arr, np.ones(n_pkts - len(arr)) * pad_value))
    
def _create_df_to_normalize_pkt_series(
    df: pd.DataFrame,
    timetofirst_colname: str,
    pkts_size_colname: str,
    pkts_dir_colname: str,
    max_n_pkts=10,
):
    max_n_pkts = int(max_n_pkts)
    series_iat = df[timetofirst_colname].apply(
        lambda l: [0] + [j - i for i, j in zip(l[:-1], l[1:])]
    )
    
    # df_pkt_size = pd.DataFrame(df[pkts_size_colname].str[:max_n_pkts].tolist())
    # df_iat = pd.DataFrame(series_iat.str[:max_n_pkts].tolist())
    # # df_iat = pd.DataFrame(series_iat.to_list())
    # df_pkt_dir = pd.DataFrame(df[pkts_dir_colname].str[:max_n_pkts].tolist())
    
    df_pkt_size = pd.DataFrame(df[pkts_size_colname].apply(_clip_and_pad, n_pkts=max_n_pkts, pad_value=0).tolist())
    df_iat = pd.DataFrame(series_iat.apply(_clip_and_pad, n_pkts=max_n_pkts, pad_value=0).tolist())
    df_pkt_dir = pd.DataFrame(df[pkts_dir_colname].apply(_clip_and_pad, n_pkts=max_n_pkts, pad_value=0).tolist())
    
    df_normalize = df_pkt_size.merge(
        df_iat, left_index=True, right_index=True, suffixes=("_size", "_iat")
    )
    df_normalize = df_normalize.merge(
        df_pkt_dir, left_index=True, right_index=True, suffixes=("", "_dir")
    )
    df_normalize.columns = [str(i) for i in range(len(df_normalize.columns))]
    return df_normalize


class FlowpicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: str | pd.DataFrame,
        timetofirst_colname: str,
        pkts_size_colname: str,
        pkts_dir_colname: str,
        target_colname: str,
        flowpic_dim: int = 32,
        flowpic_block_duration: int = 15,
        quiet: bool = False,
        logger: logging.Logger = None,
        n_workers: int = 10,
        flow_representation: MODELING_INPUT_REPR_TYPE = MODELING_INPUT_REPR_TYPE.FLOWPIC,  # str='flowpic',
        max_n_pkts: int = 10,
    ):
        """
        Arguments:
            data: if a string, it corresponds to a parquet file from
                where to load the raw data; if a pandas DataFrame,
                the data to use for the dataset
            timetofirst_colname: the column name mapping to the
                packet timeseries of containing timestamps
                (relative to the first packet of the time series)
            pkts_size_colname: the column name mapping to the
                packet size time series
            pkts_dir_colname: the column name mapping to the
                packet direction time series
            target_colname: the column name where the labels are stored
            flowpic_dim: the flowpic resolution to use
            flowpic_block_duration: how many seconds of the
                input data need to be used to generate a flowpic
            quiet: if False, no output on the console is generated when loading
            logger: the logger to use
            n_workers: how many processes to spawn when processing the data
            flow_representation: flow is represented either by "flowpic" or "pktseries", i.e. three series with pkts_size, interarrival time (derived from timetofirst) and pkt direction
            max_n_pkts: timeseries length in case of flow_representation=="pktseries"
        """
        self.scaler = None
        self.flow_representation = flow_representation
        self.max_n_pkts = max_n_pkts
        self.logger = logger
        self.timetofirst_colname = timetofirst_colname
        self.pkts_size_colname = pkts_size_colname
        self.pkts_dir_colname = pkts_dir_colname
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )
        self.n_workers = n_workers

        self.df = data
        if isinstance(data, (str, pathlib.Path)):
            if not quiet:
                self.log_msg(f"loading: {data}")
            self.df = pd.read_parquet(data)
        if "flowpic" not in self.df:
            self.df = self.add_flowpic(
                self.df,
                timetofirst_colname,
                pkts_size_colname,
                flowpic_dim,
                flowpic_block_duration,
                n_workers,
            )

        self.data = self.df["flowpic"].values
        self.target = self.df[target_colname].cat.codes.astype("int64").values
        self.target_colname = target_colname
        self.num_classes = self.df[target_colname].nunique()

    def log_msg(self, msg: str) -> None:
        """An utility function to log messages"""
        utils.log_msg(msg, self.logger)

    def set_scaler(self, scaler):
        self.scaler = scaler

    @classmethod
    def add_flowpic(
        cls,
        df: pd.DataFrame,
        timetofirst_colname: str,
        pkts_size_colname: str,
        flowpic_dim: int = 32,
        flowpic_block_duration: int = 15,
        n_workers: int = 1,
    ) -> pd.DataFrame:
        """
        Process a raw dataframe to create flowpic representation

        Arguments:
            df: a pandas DataFrame, the data to use for the dataset
            timetofirst_colname: the column name mapping to the
                packet timeseries of containing timestamps
                (relative to the first packet of the time series)
            pkts_size_colname: the column name mapping to the
                packet size time series
            flowpic_dim: the flowpic resolution to use
            flowpic_block_duration: how many seconds of the
                input data need to be used to generate a flowpic
            n_workers: how many processes to spawn when processing the data
        """
        func = functools.partial(
            augmentation.get_flowpic,
            dim=flowpic_dim,
            max_block_duration=flowpic_block_duration,
        )

        params = []
        for idx in range(df.shape[0]):
            ser = df.iloc[idx]
            params.append((ser[timetofirst_colname], ser[pkts_size_colname]))

        if n_workers > 1:
            with multiprocessing.Pool(n_workers) as pool:
                flowpic_l = pool.starmap(func, params)
        else:
            flowpic_l = [func(*pars) for pars in params]

        return df.assign(flowpic=flowpic_l)

    def __len__(self) -> int:
        """Returns how many samples are in the dataset"""
        return len(self.target)

    def __getitem__(self, index:int) -> Any:
        """
        Arguments:
            index: the index of the sample

        Return:
            A tuple with the flowpic representation (as tensor)
            and the associated label (in case flow_representation=="flowpic") or flattened, normalized timeseries (in case flow_representation=="pktseries")
        """
        if self.flow_representation == MODELING_INPUT_REPR_TYPE.FLOWPIC:  #'flowpic':
            return (
                self.transform(np.expand_dims(self.data[index], 2)).double(),
                self.target[index],
            )
        ser = self.df.iloc[[index]]
        normalize_df = _create_df_to_normalize_pkt_series(
            ser,
            self.timetofirst_colname,
            self.pkts_size_colname,
            self.pkts_dir_colname,
            self.max_n_pkts,
        )
        normalized = self.scaler.transform(normalize_df)
        return (normalized, self.target[index])

    #        df = _create_df_to_normalize_pkt_series(ser,
    #                                        self.timetofirst_colname,
    #                                        self.pkts_size_colname,
    #                                        self.pkts_dir_colname,
    #                                        self.max_n_pkts)
    #
    #        if self.scaler:
    #            df = self.scaler.transform(df)
    #        return (df, self.target[index])

    def num_classes(self):
        """Returns the number of classes in the dataset"""
        return self.df[self.target_colname].nunique()

    def samples_count(self) -> pd.Series:
        """
        Return:
            a pd.Series with the frequency count of
            number of samples per class in the dataset
        """
        return self.df[self.target_colname].value_counts()


class AugmentWhenLoadingDataset(FlowpicDataset):
    """Wrapper around FlowpicDataset to enable creation
    of augmented samples when instanciating the class
    """

    def __init__(
        self,
        data: str | pd.DataFrame,
        timetofirst_colname: str,
        pkts_size_colname: str,
        pkts_dir_colname: str,
        target_colname: str,
        flowpic_dim: int=32,
        flowpic_block_duration: int=15,
        aug_name:str ="noaug",
        aug_hparams: Dict[str, Any]=None,
        aug_samples: int=10,
        quiet: bool=False,
        logger: logging.Logger =None,
        n_workers: int=10,
        seed: int=12345,
        flow_representation:MODELING_INPUT_REPR_TYPE =MODELING_INPUT_REPR_TYPE.FLOWPIC, 
        max_n_pkts:int=10,
    ):
        """
        Arguments:
            data: if a string, it corresponds to a parquet file from
                where to load the raw data; if a pandas DataFrame,
                the data to use for the dataset
            timetofirst_colname: the column name mapping to the
                packet timeseries of containing timestamps
                (relative to the first packet of the time series)
            pkts_size_colname: the column name mapping to the
                packet size time series
            target_colname: the column name where the labels are stored
            flowpic_dim: the flowpic resolution to use
            flowpic_block_duration: how many seconds of the
                input data need to be used to generate a flowpic
            aug_name: one of {"noaug", "rotate", "horizontalflip",
                "colorjitter", "packetloss", "changertt", "timeshift"
            aug_hparams: the augmentation parameters (see the augmentation module)
            aug_samples: how many samples to create for each
                already existing sample
            quiet: if False, no output on the console is generated when loading
            logger: the logger to use
            n_workers: how many processes to spawn when processing the data
            seed: random seed
            flow_representation: a MODELING_INPUT_REPR_TYPE value
            max_n_pkts: packet series len (if flow_representation == MODELING_INPUT_REPR_TYPE.PKTSERIES)
        """
        super().__init__(
            data,
            timetofirst_colname=timetofirst_colname,
            pkts_size_colname=pkts_size_colname,
            pkts_dir_colname=pkts_dir_colname,
            target_colname=target_colname,
            flowpic_dim=flowpic_dim,
            flowpic_block_duration=flowpic_block_duration,
            quiet=quiet,
            logger=logger,
            n_workers=n_workers,
            flow_representation=flow_representation,
            max_n_pkts=max_n_pkts,
        )
        self.aug_samples = aug_samples
        self.seed = seed

        self.df = self.samples_augmentation(
            aug_name=aug_name,
            aug_hparams=aug_hparams,
            samples=aug_samples,
            flowpic_dim=flowpic_dim,
            flowpic_block_duration=flowpic_block_duration,
            seed=seed,
        )

        self.data = self.df["flowpic"].values
        self.target = self.df[target_colname].cat.codes.astype("int64").values
        self.target_colname = target_colname
        self.num_classes = self.df[target_colname].nunique()

    def regenerate_flowpic(
        self, dim: int = 32, block_duration: int = 15
    ) -> pd.DataFrame:
        """
        Overwrite the existing flowpic by creating new ones

        Arguments:
            dim: the flowpic resolution
            block_duration: the max number of seconds for each sample time series
                to consider when computing a flowpic

        Return:
            A new dataframe with a "flowpic" column reporting the flowpic
            for each available sample
        """
        self.df = self.add_flowpic(
            self.df,
            self.timetofirst_colname,
            self.pkts_size_colname,
            dim,
            block_duration,
            self.n_workers,
        )
        return self.df

    def _worker_aug_torch(
        self,
        df: pd.DataFrame,
        aug_class: augmentation.Augmentation,
        samples: int = 9,
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Helper function for a multi-processing worker handling
        augmentations related to flowpic (i.e., working on Tensor data)

        Arguments:
            df: a batch of samples to process
            aug_class: an instance of an augmentation class (see augmentation module)
            samples: how many samples to create for each existings sample

        Return:
            An expanded version of the input dataframe containing
            the new samples. IMPORTANT: the new DataFrame is performing
            a shallow copy of the original and act only of a subset of
            columns. Hence, the returned version might have semantical
            incosistencies
        """
        new_flowpic = []
        new_aug_params = []
        df = pd.concat([df.copy() for _ in range(samples)]).assign(
            is_augmented=True, aug_params={}
        )
        dtypes = dict(self.df.dtypes)
        for idx in range(df.shape[0]):
            ser = df.iloc[idx]
            new_sample = ser.copy()
            new_flowpic.append(aug_class(new_sample["flowpic"]))
            new_aug_params.append(aug_class.get_params().copy())
        print(".", end="", flush=True)
        df = df.assign(flowpic=new_flowpic, aug_params=new_aug_params)
        return df

    def _worker_aug_numpy(
        self,
        df: pd.DataFrame,
        aug_class: augmentation.Augmentation,
        samples: int = 9,
        flowpic_dim: int = 32,
        flowpic_block_duration: int = 15,
    ):
        """
        Helper function for a multi-processing worker handling
        augmentations related to time series (i.e., working with numpy arrays)

        Arguments:
            df: a batch of samples to process
            aug_class: an instance of an augmentation class (see augmentation module)
            samples: how many samples to create for each existings sample
            flowpic_dim: the flowpic resolution
            flowpic_block_duration: the max number of seconds for each sample time series
                to consider when computing a flowpic

        Return:
            An expanded version of the input dataframe containing
            the new samples. IMPORTANT: the new DataFrame is performing
            a shallow copy of the original and act only of a subset of
            columns. Hence, the returned version might have semantical
            incosistencies
        """
        new_timetofirst = []
        new_pkts_size = []
        new_flowpic = []
        new_aug_params = []
        df = pd.concat([df.copy() for _ in range(samples)]).assign(
            is_augmented=True, aug_params={}
        )
        for idx in range(df.shape[0]):
            ser = df.iloc[idx]
            _timetofirst, _pkts_size, indexes = aug_class(
                ser[self.timetofirst_colname], ser[self.pkts_size_colname]
            )
            new_aug_params.append(aug_class.get_params().copy())
            new_timetofirst.append(_timetofirst)
            new_pkts_size.append(_pkts_size)
            new_flowpic.append(
                augmentation.get_flowpic(
                    _timetofirst, _pkts_size, flowpic_dim, flowpic_block_duration
                )
            )
        df = df.assign(
            timetofirst=new_timetofirst,
            pkts_size=new_pkts_size,
            flowpic=new_flowpic,
            aug_params=new_aug_params,
        )
        print(".", end="", flush=True)
        return df

    def _samples_augmentation_loop(
        self,
        aug_name: str,
        aug_hparams: Dict[str, Any],
        samples: int,
        worker_func: Callable,
        seed: int = 12345,
        flowpic_dim: int = 32,
        flowpic_block_duration: int = 15,
    ) -> pd.DataFrame:
        """
        Helper function handling the main loop for samples augmentation
        by means of multiprocessing

        Arguments:
            aug_name: one of {"noaug", "rotate", "horizontalflip",
                "colorjitter", "packetloss", "changertt", "timeshift"
            aug_hparams: the augmentation parameters (see the augmentation module)
            samples: how many samples to create for each
                already existing sample
            worker_func: the callback to use for augmentation
            seed: the seed to use for augmentation
            flowpic_dim: the flowpic resolution
            flowpic_block_duration: the max number of seconds for each sample time series
                to consider when computing a flowpic

        Return:
            A pandas DataFrame with all the original samples plus
            the augmented ones
        """
        if aug_hparams is None:
            aug_hparams = dict()
        params = []
        indexes = self.df.index.values
        partition_size = indexes.shape[0] // self.n_workers
        for idx in range(0, len(indexes), partition_size):
            rng = np.random.default_rng(seed + idx)
            # aug_class = augmentation.augmentation_factory(aug_name, rng, **aug_hparams)
            aug_class = augmentation.augmentation_factory(aug_name, rng, aug_hparams)
            partition_indexes = indexes[idx : idx + partition_size]
            params.append(
                (
                    self.df.loc[partition_indexes],
                    aug_class,
                    samples,
                    flowpic_dim,
                    flowpic_block_duration,
                )
            )

        # Note: this is a very dirty trick
        #
        # We experienced deadlocks similar to what reported here
        # https://github.com/pytorch/pytorch/issues/3492
        # when using torchvision.transforms (with both functional APIs
        # and instanciating classes). But the logic in the
        # .augmentation module works fine in single process
        #
        # Relying on torch.multiprocessing
        # https://github.com/pytorch/pytorch/issues/3492
        # and setting torch.multiprogressing.set_start_method('spawn')
        # (in if __name__ == '__main__') fixed the issue
        #
        # But this requires invoking the .Pool() differently
        # depending on the underlining augmentation (pytorch or numpy)
        if worker_func.__name__ == "_worker_aug_torch":
            with torch.multiprocessing.Pool(self.n_workers) as pool:
                augmented_l = pool.starmap(worker_func, params)
        else:
            with multiprocessing.Pool(self.n_workers) as pool:
                augmented_l = pool.starmap(worker_func, params)

        self.df = pd.concat([self.df] + augmented_l).reset_index()
        return self.df

    def samples_augmentation(
        self,
        aug_name:str="noaug",
        aug_hparams:Dict[str, Any]=None,
        samples:int=None,
        seed:int=12345,
        flowpic_dim:int=32,
        flowpic_block_duration:int=15,
    ):
        """Applies samples augmentation

        Arguments:
            aug_name: one of {"rotate", "horizontalflip", "colorjitter", "packetloss", "timeshift", "changertt" or "noaug"}
            aug_hparams: the augmentation parameters (see the augmentation module)
            samples: final number of samples for each individual sample (e.g., samples=10 means the original sample and 9 augmented versions)
            seed: random number generator seed
            flowpic_dim: the flowpic resolution
            flowpic_block_duration: the max number of seconds for each sample time series
                to consider when computing a flowpic

        Return:
            A pandas DataFrame with all original samples and the augmented ones
        """
        if samples is None:
            samples = self.aug_samples

        if aug_name not in augmentation.AUGMENTATION_CLASSES:
            self.log_msg("no augmentation")
            return self.df.assign(is_augmented=False)

        self.df = self.df.assign(is_augmented=False)
        samples -= 1
        if aug_name == "horizontalflip":
            samples = 1
        worker_func = self._worker_aug_numpy
        if aug_name in ("rotate", "horizontalflip", "colorjitter"):
            worker_func = self._worker_aug_torch

        self.log_msg(f"data augmentation ({aug_name})")
        self._samples_augmentation_loop(
            aug_name=aug_name,
            aug_hparams=aug_hparams,
            samples=samples,
            worker_func=worker_func,
            seed=seed,
            flowpic_dim=flowpic_dim,
            flowpic_block_duration=flowpic_block_duration,
        )
        print()
        return self.df


class MultiViewDataset(FlowpicDataset):
    def __init__(
        self,
        data,
        timetofirst_colname,
        pkts_size_colname,
        pkts_dir_colname,
        target_colname,
        flowpic_dim,
        flowpic_block_duration,
        quiet=False,
        logger=None,
        n_workers=10,
        seed=12345,
        aug_config=None,
        num_views=1,
        yield_also_original=False,
        **kwargs,
    ):
        super().__init__(
            data=data,
            timetofirst_colname=timetofirst_colname,
            pkts_size_colname=pkts_size_colname,
            pkts_dir_colname=pkts_dir_colname,
            target_colname=target_colname,
            flowpic_dim=flowpic_dim,
            flowpic_block_duration=flowpic_block_duration,
            quiet=quiet,
            logger=logger,
            n_workers=n_workers,
        )
        self.num_views = num_views
        self.yield_also_original = yield_also_original
        self.aug_config = OrderedDict(aug_config)
        self.seed = seed

        self.rng_aug = None
        self.aug_classes = []
        self.aug_names = []
        self.reset_augmentations(self.aug_config, seed)

    def reset_augmentations(self, aug_config=None, seed=12345):
        if aug_config is None:
            if self.aug_config is None:
                return
            aug_config = self.aug_config

        self.rng_aug = np.random.default_rng(seed)
        self.aug_config = OrderedDict(aug_config)
        self.aug_names = list(self.aug_config.keys())
        self.aug_classes = [
            augmentation.augmentation_factory(aug_name, self.rng_aug, aug_hparams)
            for aug_name, aug_hparams in aug_config.items()
        ]

    def _apply_multiple_augmentations(self, flowpic, timetofirst, pkts_size):
        kwargs = dict(
            flowpic=np.copy(flowpic),
            timetofirst=np.copy(timetofirst),
            pkts_size=np.copy(pkts_size),
        )
        for idx in self.rng_aug.permutation(len(self.aug_classes)):
            aug = self.aug_classes[idx]
            name = self.aug_names[idx]
            kwargs = augmentation.apply_augmentation(name, aug, **kwargs)
        return kwargs["flowpic"]

    def __getitem__(self, idx):
        ser = self.df.iloc[idx]
        flowpic = ser["flowpic"]
        timetofirst = ser[self.timetofirst_colname]
        pkts_size = ser[self.pkts_size_colname]

        views = [
            self.transform(
                self._apply_multiple_augmentations(flowpic, timetofirst, pkts_size)
            ).double()
            for _ in range(self.num_views)
        ]
        if self.yield_also_original:
            views.append(self.transform(flowpic).double())
        return views, self.target[idx]  # , self.worker_id, idx


def _verify_augmentation_options(aug_config, aug_samples, aug_when_loading):
    if aug_config is None:
        aug_config = dict(noaug=None)

    if aug_when_loading:
        return aug_config, aug_samples, aug_when_loading

    # enforce at most 2 positive samples for multiview
    aug_samples = min(2, aug_samples)
    if "noaug" in aug_config:
        del aug_config["noaug"]
    if len(aug_config) == 0:
        raise RuntimeError(
            f"No augmentation configured, but they are required for a multi-view dataset"
        )
    return aug_config, aug_samples, aug_when_loading


def train_test_split(
    df: pd.DataFrame,
    target_colname: str,
    timetofirst_colname: str = None,
    pkts_size_colname: str = None,
    pkts_dir_colname: str = None,
    max_samples_per_class: int = None,
    train_val_split_ratio: float = None,
    seed: int = 12345,
    flow_representation: MODELING_INPUT_REPR_TYPE = MODELING_INPUT_REPR_TYPE.FLOWPIC,  # str='flowpic',
    max_n_pkts: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #### XXXX
    if max_samples_per_class is None:
        max_samples_per_class = -1

    rng = np.random.default_rng(seed)
    indexes_train = []
    indexes_val = []
    for app in df[target_colname].unique():
        indexes = df[df[target_colname] == app].index.values
        rng.shuffle(indexes)

        samples = len(indexes)
        if max_samples_per_class > 0:
            samples = min(samples, max_samples_per_class)

        if train_val_split_ratio:
            train_samples = int(samples * train_val_split_ratio)
            indexes_train.append(indexes[:train_samples])
            indexes_val.append(indexes[train_samples:samples])
        else:
            indexes_train.append(indexes[:samples])
            indexes_val.append(indexes[samples:])

    df_train = df.loc[np.concatenate(indexes_train)]

    # if flow_representation=='pktseries':
    if (
        flow_representation == MODELING_INPUT_REPR_TYPE.PKTSERIES
    ):  # INPUT_REPRESENTATION_TYPE.PKTS_TIME_SERIES:
        df_normalize = _create_df_to_normalize_pkt_series(
            df_train,
            timetofirst_colname,
            pkts_size_colname,
            pkts_dir_colname,
            max_n_pkts,
        )
        scaler = MinMaxScaler()
        scaler.fit(df_normalize)
    else:
        scaler = None

    indexes_val = np.concatenate(indexes_val)
    if len(indexes_val) == 0:
        raise RuntimeError(f"Invalid dataset: the validation set cannot be empty")
    df_val = df.loc[indexes_val]
    return df_train, df_val, scaler


def load_ucdavis_icdm19_train_val_datasets(
    # folder,
    split_idx=0,
    flowpic_dim=32,
    flowpic_block_duration=15,
    max_samples_per_class=-1,
    aug_config=None,
    aug_samples=10,
    aug_when_loading=True,
    aug_yield_also_original=False,
    seed=12345,
    n_workers=50,
    quiet=False,
    logger=None,
    train_val_split_ratio: float = 0.8,
    suppress_val_augmentation: bool = False,
    flow_representation: MODELING_INPUT_REPR_TYPE = MODELING_INPUT_REPR_TYPE.FLOWPIC,  # str='flowpic',
    max_n_pkts: int = 10,
    **kwargs,
) -> Tuple[FlowpicDataset, FlowpicDataset]:
    aug_config, aug_samples, aug_when_loading = _verify_augmentation_options(
        aug_config, aug_samples, aug_when_loading
    )
    # folder = pathlib.Path(folder)
    if split_idx is not None and split_idx < 0:
        split_idx = None

    if split_idx is not None and split_idx >= 0:
        # fname = folder / 'imc23' / f"train_split_{split_idx}.parquet"
        df = tcbench.load_parquet(
            tcbench.DATASETS.UCDAVISICDM19, split=split_idx, animation=True
        )
        fname = datasets_utils.get_dataset_parquet_filename(
            tcbench.DATASETS.UCDAVISICDM19, split=split_idx
        )
        utils.log_msg(f"loaded: {fname}", logger)
        # df = pd.read_parquet(fname)
    else:
        # fname = folder / 'ucdavis-icdm19.parquet'
        df = tcbench.load_parquet(
            tcbench.DATASETS.UCDAVISICDM19, min_pkts=-1, split=None, animation=True
        )
        fname = datasets_utils.get_dataset_parquet_filename(
            tcbench.DATASETS.UCDAVISICDM19, min_pkts=-1, split=None
        )
        utils.log_msg(f"loaded: {fname}", logger)
        df = pd.read_parquet(fname)
        # we need to select only the pretraining partition
        # as the other two are specific for testing
        df = df[df["partition"] == "pretraining"]

    # Quote from Sec.3.2
    # > We also train without any augmentation as
    # > baseline experiments and term it "no aug".
    # > For all experiments we allocated 20% of the
    # > images for validation, and early stopped the
    # training when the validation loss stopped improving.
    timetofirst_colname = "timetofirst"
    pkts_size_colname = "pkts_size"
    pkts_dir_colname = "pkts_dir"
    df_train, df_val, scaler = train_test_split(
        df,
        timetofirst_colname=timetofirst_colname,
        pkts_size_colname=pkts_size_colname,
        pkts_dir_colname=pkts_dir_colname,
        target_colname="app",
        max_samples_per_class=max_samples_per_class,
        train_val_split_ratio=train_val_split_ratio,
        flow_representation=flow_representation,
        max_n_pkts=max_n_pkts,
        seed=seed,
    )

    dsetclass = AugmentWhenLoadingDataset
    params = dict(
        timetofirst_colname=timetofirst_colname,
        pkts_size_colname=pkts_size_colname,
        pkts_dir_colname=pkts_dir_colname,
        target_colname="app",
        flowpic_dim=flowpic_dim,
        flowpic_block_duration=flowpic_block_duration,
        quiet=quiet,
        logger=logger,
        n_workers=n_workers,
        seed=seed,
        flow_representation=flow_representation,
        max_n_pkts=max_n_pkts,
    )

    train_params = deepcopy(params)
    val_params = deepcopy(params)
    if aug_when_loading:
        # we pick only one augmentation...because there should
        # be only one provided in the input
        aug_name, aug_hparams = list(aug_config.items())[0]
        specific_params = dict(
            aug_name=aug_name,
            aug_hparams=aug_hparams,
            aug_samples=aug_samples,
        )
        train_params.update(specific_params)

        val_params = deepcopy(train_params)
        if suppress_val_augmentation:
            # by default, validation is augmented as well based on
            # the following quote from the IMC22 paper
            # > For all experiments, we apply each of the augmentations
            # > 10 times on the 100 samples per class training set, which
            # > increase the training set to 1000 images per class.
            # > We also train without any augmentation as baseline
            # > experiments and term it "no aug". For all experiments we allocated
            # > 20% of the images for validation, and early stopped the
            # > training when the validation loss stopped improving
            specific_params = dict(
                aug_name="noaug",
                aug_hparams=None,
                aug_samples=None,
            )
            val_params.update(specific_params)

    else:
        dsetclass = MultiViewDataset
        specific_params = dict(
            aug_config=aug_config,
            num_views=aug_samples,
            yield_also_original=aug_yield_also_original,
            seed=seed,
        )
        # both train and val need to be augmented
        train_params.update(specific_params)
        val_params.update(specific_params)

    dset_train = dsetclass(data=df_train, **train_params)
    dset_train.set_scaler(scaler)
    dset_val = dsetclass(data=df_val, **val_params)
    dset_val.set_scaler(scaler)

    if not quiet:
        utils.log_msg("dataset samples count", logger)
        df_samples = pd.DataFrame(
            [dset_train.samples_count(), dset_val.samples_count()],
            index=["train", "val"],
        ).T
        utils.log_msg(df_samples, logger)
    return dset_train, dset_val


def load_ucdavis_icdm19_test_dataset(
    # folder:pathlib.Path,
    flowpic_dim: int = 32,
    flowpic_block_duration: int = 15,
    quiet: bool = False,
    logger: logging.Logger = None,
    flow_representation: MODELING_INPUT_REPR_TYPE = MODELING_INPUT_REPR_TYPE.FLOWPIC,  #'flowpic',
    max_n_pkts=10,
    **kwargs,
) -> Dict[str, FlowpicDataset]:
    # folder = pathlib.Path(folder)
    params = dict(
        timetofirst_colname="timetofirst",
        pkts_size_colname="pkts_size",
        pkts_dir_colname="pkts_dir",
        target_colname="app",
        flowpic_dim=flowpic_dim,
        flowpic_block_duration=flowpic_block_duration,
        quiet=quiet,
        logger=logger,
        flow_representation=flow_representation,
        max_n_pkts=max_n_pkts,
    )
    # dset_human = FlowpicDataset(data=folder/'imc23' / "test_split_human.parquet", **params)
    # dset_script = FlowpicDataset(data=folder/'imc23' / "test_split_script.parquet", **params)
    dset_human = FlowpicDataset(
        data=datasets_utils.get_dataset_parquet_filename(
            tcbench.DATASETS.UCDAVISICDM19, split="human"
        ),
        **params,
    )
    dset_script = FlowpicDataset(
        data=datasets_utils.get_dataset_parquet_filename(
            tcbench.DATASETS.UCDAVISICDM19, split="script"
        ),
        **params,
    )

    if not quiet:
        df_samples = pd.DataFrame(
            [
                dset_human.samples_count(),
                dset_script.samples_count(),
            ],
            index=["human", "script"],
        ).T
        utils.log_msg(df_samples, logger)

    dsets = dict(
        human=dset_human,
        script=dset_script,
    )
    return dsets


def load_ucdavis_icdm19_for_finetuning_dataset(
    # folder:pathlib.Path,
    flowpic_dim: int = 32,
    flowpic_block_duration: int = 15,
    quiet: bool = False,
    logger: logging.Logger = None,
    train_samples: int = 10,
    aug_config=None,
    aug_samples=10,
    aug_when_loading=True,
    aug_yield_also_original=False,
    seed: int = 12345,
    dataset_minpkts=-1,
    **kwargs,
) -> Dict[str, FlowpicDataset]:
    dset_dict = load_ucdavis_icdm19_test_dataset(
        # folder,
        flowpic_dim=flowpic_dim,
        flowpic_block_duration=flowpic_block_duration,
        quiet=True,
        logger=logger,
    )

    samples_count = []
    samples_colnames = []
    new_dset_dict = dict()
    params = dict(
        timetofirst_colname="timetofirst",
        pkts_size_colname="pkts_size",
        pkts_dir_colname="pkts_dir",
        target_colname="app",
        flowpic_dim=flowpic_dim,
        flowpic_block_duration=flowpic_block_duration,
        logger=logger,
    )
    for name, dset in dset_dict.items():
        df = dset.df
        df_train, df_test, scaler = train_test_split(
            df, target_colname="app", max_samples_per_class=train_samples, seed=seed
        )
        train_name = f"{name}_train"
        test_name = f"{name}_test"

        # test set is a basic FlowpicDataset
        new_dset_dict[test_name] = FlowpicDataset(data=df_test, **params)

        # when passing augmentation policy, the training
        # dataset is augmented with multi-view
        if aug_config is None:
            new_dset_dict[train_name] = FlowpicDataset(data=df_train, **params)
        else:
            specific_params = dict(
                aug_config=aug_config,
                num_views=aug_samples,
                yield_also_original=aug_yield_also_original,
                seed=seed,
            )
            new_dset_dict[train_name] = MultiViewDataset(
                data=df_train, **params, **specific_params
            )

        samples_count.append(new_dset_dict[train_name].samples_count())
        samples_count.append(new_dset_dict[test_name].samples_count())
        samples_colnames.append(train_name)
        samples_colnames.append(test_name)

    if not quiet:
        df_samples = pd.DataFrame(samples_count, index=samples_colnames).T
        df_samples = df_samples[sorted(df_samples.columns.tolist())]
        utils.log_msg(df_samples, logger)
    return new_dset_dict


def load_ucdavis_icdm19_train_val_leftover_dataset(
    # folder:pathlib.Path,
    flowpic_dim: int = 32,
    flowpic_block_duration: int = 15,
    quiet: bool = False,
    logger: logging.Logger = None,
    dset_train: FlowpicDataset = None,
    dset_val: FlowpicData = None,
    flow_representation: MODELING_INPUT_REPR_TYPE = MODELING_INPUT_REPR_TYPE.FLOWPIC,  # str='flowpic',
    max_n_pkts: int = 10,
    **kwargs,
) -> FlowpicDataset:
    indexes = []

    #        folder = pathlib.Path(folder)
    #
    #        fname = folder / 'ucdavis-icdm19.parquet'
    #        utils.log_msg(f'loading: {fname}', logger)
    #        df = pd.read_parquet(folder / 'ucdavis-icdm19.parquet')
    df = datasets_utils.load_parquet(
        tcbench.DATASETS.UCDAVISICDM19, min_pkts=-1, split=None, animation=True
    )
    path = datasets_utils.get_dataset_parquet_filename(
        tcbench.DATASETS.UCDAVISICDM19, min_pkts=-1, split=None
    )
    utils.log_msg(f"loaded: {path}", logger)

    indexes = set(df["row_id"].values)
    if dset_train:
        indexes -= set(dset_train.df["row_id"].values)
    if dset_val:
        indexes -= set(dset_val.df["row_id"].values)

    df_leftover = df[df["row_id"].isin(indexes)]
    dset = FlowpicDataset(
        data=df_leftover,
        timetofirst_colname="timetofirst",
        pkts_size_colname="pkts_size",
        pkts_dir_colname="pkts_dir",
        target_colname="app",
        flowpic_dim=flowpic_dim,
        flowpic_block_duration=flowpic_block_duration,
        logger=logger,
        flow_representation=flow_representation,
        max_n_pkts=max_n_pkts,
    )
    return dset


def load_utmobilenet21_train_val_datasets(
    # folder,
    split_idx=0,
    flowpic_dim=32,
    flowpic_block_duration=15,
    max_samples_per_class=-1,
    aug_config=None,
    aug_samples=10,
    aug_when_loading=True,
    aug_yield_also_original=False,
    seed=12345,
    n_workers=10,
    quiet=False,
    logger=None,
    train_val_split_ratio: float = 0.8,
    suppress_val_augmentation: bool = False,
    dataset_minpkts=-1,
) -> Tuple[FlowpicDataset, FlowpicDataset]:
    aug_config, aug_samples, aug_when_loading = _verify_augmentation_options(
        aug_config, aug_samples, aug_when_loading
    )

    #    folder = pathlib.Path(folder)
    #
    #    prefix = 'utmobilenet21_filtered'
    #    if dataset_minpkts != -1:
    #        prefix += f'_minpkts{dataset_minpkts}'
    #
    #    fname = folder / 'imc23' / f'{prefix}.parquet'
    #    utils.log_msg(f'loading: {fname}', logger)
    #    df = pd.read_parquet(fname)
    #    PRESELECTED_COLUMNS = [
    #        timetofirst_colname,
    #        pkts_size_colname,
    #        pkts_dir_colname,
    #        target_colname,
    #        row_id_colname
    #    ]

    dataset_name = tcbench.DATASETS.UTMOBILENET21
    df = datasets_utils.load_parquet(
        dataset_name, min_pkts=dataset_minpkts, split=None, animation=True
    )
    path = datasets_utils.get_dataset_parquet_filename(
        dataset_name, min_pkts=dataset_minpkts, split=None
    )
    utils.log_msg(f"loaded: {path}", logger)

    if max_samples_per_class == -1:
        # fname = folder / 'imc23' / f'{prefix}_splits.parquet'
        # utils.log_msg(f'loading: {fname}', logger)
        # df_splits = pd.read_parquet(fname)
        df_splits = datasets_utils.load_parquet(
            dataset_name, min_pkts=dataset_minpkts, split=True, animation=True
        )
        path = datasets_utils.get_dataset_parquet_filename(
            dataset_name, min_pkts=dataset_minpkts, split=True
        )
        utils.log_msg(f"loaded: {path}", logger)

        ser = df_splits[df_splits["split_index"] == split_idx].iloc[0]
        train_indexes = ser["train_indexes"]
        val_indexes = ser["val_indexes"]
    else:
        df_train, df_val, scaler = train_test_split(
            df,
            timetofirst_colname="timetofirst",
            pkts_size_colname="pkts_size",
            target_colname="app",
            max_samples_per_class=max_samples_per_class,
            train_val_split_ratio=train_val_split_ratio,
            seed=seed,
        )
        train_indexes = df_train["row_id"].values
        val_indexes = df_val["row_id"].values

    dsetclass = AugmentWhenLoadingDataset
    params = dict(
        timetofirst_colname="timetofirst",
        pkts_size_colname="pkts_size",
        pkts_dir_colname="pkts_dir",
        target_colname="app",
        flowpic_dim=flowpic_dim,
        flowpic_block_duration=flowpic_block_duration,
        quiet=quiet,
        logger=logger,
        n_workers=n_workers,
        seed=seed,
    )

    train_params = deepcopy(params)
    val_params = deepcopy(params)
    if aug_when_loading:
        # we pick only one augmentation...because there should
        # be only one provided in the input
        aug_name, aug_hparams = list(aug_config.items())[0]
        specific_params = dict(
            aug_name=aug_name,
            aug_hparams=aug_hparams,
            aug_samples=aug_samples,
        )
        train_params.update(specific_params)

        val_params = deepcopy(train_params)
        if suppress_val_augmentation:
            # by default, validation is augmented as well based on
            # the following quote from the IMC22 paper
            # > For all experiments, we apply each of the augmentations
            # > 10 times on the 100 samples per class training set, which
            # > increase the training set to 1000 images per class.
            # > We also train without any augmentation as baseline
            # > experiments and term it "no aug". For all experiments we allocated
            # > 20% of the images for validation, and early stopped the
            # > training when the validation loss stopped improving
            specific_params = dict(
                aug_name="noaug",
                aug_hparams=None,
                aug_samples=None,
            )
            val_params.update(specific_params)

    else:
        dsetclass = MultiViewDataset
        specific_params = dict(
            aug_config=aug_config,
            num_views=aug_samples,
            yield_also_original=aug_yield_also_original,
        )
        # both train and val need to be augmented
        train_params.update(specific_params)
        val_params.update(specific_params)

    df_train = df.loc[train_indexes]
    df_val = df.loc[val_indexes]
    dset_train = dsetclass(data=df_train, **train_params)
    dset_val = dsetclass(data=df_val, **val_params)

    if not quiet:
        utils.log_msg("dataset samples count", logger)
        df_samples = pd.DataFrame(
            [dset_train.samples_count(), dset_val.samples_count()],
            index=["train", "val"],
        ).T
        utils.log_msg(df_samples, logger)

    return dset_train, dset_val


def load_utmobilenet21_test_dataset(
    # folder:pathlib.Path,
    split_idx: int = 0,
    flowpic_dim: int = 32,
    flowpic_block_duration: int = 15,
    quiet: bool = False,
    logger: logging.Logger = None,
    dataset_minpkts=-1,
) -> Dict[str, FlowpicDataset]:
    #    folder = pathlib.Path(folder)
    #
    #    prefix = 'utmobilenet21_filtered'
    #    if dataset_minpkts != -1:
    #        prefix += f'_minpkts{dataset_minpkts}'
    #
    #    fname = folder / 'imc23' / f'{prefix}.parquet'
    #    utils.log_msg(f'loading: {fname}', logger)
    #    df = pd.read_parquet(fname)
    #
    #    fname = folder / 'imc23' / f'{prefix}_splits.parquet'
    #    utils.log_msg(f'loading: {fname}', logger)
    #    df_splits = pd.read_parquet(fname)

    df = datasets_utils.load_parquet(
        tcbench.DATASETS.UTMOBILENET21, dataset_minpkts, None, animation=True
    )
    path = datasets_utils.get_dataset_parquet_filename(
        tcbench.DATASETS.UTMOBILENET21, dataset_minpkts, None
    )
    utils.log_msg(f"loaded: {path}")

    df_splits = datasets_utils.load_parquet(
        tcbench.DATASETS.UTMOBILENET21, dataset_minpkts, split_idx, animation=True
    )
    path = datasets_utils.get_dataset_parquet_filename(
        tcbench.DATASETS.UTMOBILENET21, dataset_minpkts, split_idx
    )
    utils.log_msg(f"loaded: {path}")

    ser = df_splits[df_splits["split_index"] == split_idx].iloc[0]
    test_indexes = ser["test_indexes"]
    df_test = df.loc[test_indexes]

    params = dict(
        timetofirst_colname="timetofirst",
        pkts_size_colname="pkts_size",
        pkts_dir_colname="pkts_dir",
        target_colname="app",
        flowpic_dim=flowpic_dim,
        flowpic_block_duration=flowpic_block_duration,
        quiet=quiet,
        logger=logger,
    )
    dset = FlowpicDataset(data=df_test, **params)

    if not quiet:
        utils.log_msg(dset.samples_count(), logger)

    name = "test"
    dsets = {name: dset}
    return dsets


def generic_load_train_val_datasets(
    # folder,
    dataset_name,
    dataset_minpkts: int = -1,
    split_idx=0,
    flowpic_dim=32,
    flowpic_block_duration=15,
    max_samples_per_class=-1,
    aug_config=None,
    aug_samples=10,
    aug_when_loading=True,
    aug_yield_also_original=False,
    seed=12345,
    n_workers=10,
    quiet=False,
    logger=None,
    train_val_split_ratio: float = 0.8,
    suppress_val_augmentation: bool = False,
    timetofirst_colname="timetofirst",
    pkts_size_colname="pkts_size",
    pkts_dir_colname="pkts_dir",
    target_colname="app",
    row_id_colname="row_id",
    load_all_columns=False,
) -> Tuple[FlowpicDataset, FlowpicDataset]:
    aug_config, aug_samples, aug_when_loading = _verify_augmentation_options(
        aug_config, aug_samples, aug_when_loading
    )

    #    folder = pathlib.Path(folder)
    #
    #    prefix = f'{dataset_name}_filtered'
    #    if dataset_minpkts != -1:
    #        prefix = f'{prefix}_minpkts{dataset_minpkts}'
    #
    #    fname = folder / 'imc23' / f'{prefix}.parquet'
    #    utils.log_msg(f'loading: {fname}', logger)
    #
    #    if load_all_columns:
    #        df = pd.read_parquet(fname)
    #    else:
    #        df = pd.read_parquet(fname, columns=[timetofirst_colname, pkts_size_colname, pkts_dir_colname, target_colname, row_id_colname])
    PRESELECTED_COLUMNS = [
        timetofirst_colname,
        pkts_size_colname,
        pkts_dir_colname,
        target_colname,
        row_id_colname,
    ]
    columns = None if load_all_columns else PRESELECTED_COLUMNS

    df = datasets_utils.load_parquet(
        dataset_name,
        min_pkts=dataset_minpkts,
        split=None,
        columns=columns,
        animation=True,
    )
    path = datasets_utils.get_dataset_parquet_filename(
        dataset_name, min_pkts=dataset_minpkts, split=None
    )
    utils.log_msg(f"loaded: {path}", logger)

    if max_samples_per_class == -1:
        ##fname = folder / 'imc23' / f'{prefix}_100samples_splits.parquet'
        # fname = folder / 'imc23' / f'{prefix}_splits.parquet'
        # utils.log_msg(f'loading: {fname}', logger)

        # df_splits = pd.read_parquet(fname)
        df_splits = datasets_utils.load_parquet(
            dataset_name, min_pkts=dataset_minpkts, split=True, animation=True
        )
        path = datasets_utils.get_dataset_parquet_filename(
            dataset_name, min_pkts=dataset_minpkts, split=True
        )
        utils.log_msg(f"loaded: {path}", logger)

        ser = df_splits[df_splits["split_index"] == split_idx].iloc[0]
        train_indexes = ser["train_indexes"]
        val_indexes = ser["val_indexes"]
    else:
        df_train, df_val, _ = train_test_split(
            df,
            timetofirst_colname="timetofirst",
            pkts_size_colname="pkts_size",
            target_colname="app",
            max_samples_per_class=max_samples_per_class,
            train_val_split_ratio=train_val_split_ratio,
            seed=seed,
        )
        train_indexes = df_train["row_id"].values
        val_indexes = df_val["row_id"].values

    dsetclass = AugmentWhenLoadingDataset
    params = dict(
        timetofirst_colname=timetofirst_colname,
        pkts_size_colname=pkts_size_colname,
        pkts_dir_colname=pkts_dir_colname,
        target_colname=target_colname,
        flowpic_dim=flowpic_dim,
        flowpic_block_duration=flowpic_block_duration,
        quiet=quiet,
        logger=logger,
        n_workers=n_workers,
        seed=seed,
    )

    train_params = deepcopy(params)
    val_params = deepcopy(params)
    if aug_when_loading:
        # we pick only one augmentation...because there should
        # be only one provided in the input
        aug_name, aug_hparams = list(aug_config.items())[0]
        specific_params = dict(
            aug_name=aug_name,
            aug_hparams=aug_hparams,
            aug_samples=aug_samples,
        )
        train_params.update(specific_params)

        val_params = deepcopy(train_params)
        if suppress_val_augmentation:
            # by default, validation is augmented as well based on
            # the following quote from the IMC22 paper
            # > For all experiments, we apply each of the augmentations
            # > 10 times on the 100 samples per class training set, which
            # > increase the training set to 1000 images per class.
            # > We also train without any augmentation as baseline
            # > experiments and term it "no aug". For all experiments we allocated
            # > 20% of the images for validation, and early stopped the
            # > training when the validation loss stopped improving
            specific_params = dict(
                aug_name="noaug",
                aug_hparams=None,
                aug_samples=None,
            )
            val_params.update(specific_params)

    else:
        dsetclass = MultiViewDataset
        specific_params = dict(
            aug_config=aug_config,
            num_views=aug_samples,
            yield_also_original=aug_yield_also_original,
        )
        # both train and val need to be augmented
        train_params.update(specific_params)
        val_params.update(specific_params)

    df_train = df.loc[train_indexes]
    df_val = df.loc[val_indexes]
    dset_train = dsetclass(data=df_train, **train_params)
    dset_val = dsetclass(data=df_val, **val_params)

    if not quiet:
        utils.log_msg("dataset samples count", logger)
        df_samples = pd.DataFrame(
            [dset_train.samples_count(), dset_val.samples_count()],
            index=["train", "val"],
        ).T
        utils.log_msg(df_samples, logger)

    return dset_train, dset_val


def generic_load_test_dataset(
    # folder:pathlib.Path,
    dataset_name: str,
    dataset_minpkts: int = -1,
    split_idx: int = 0,
    flowpic_dim: int = 32,
    flowpic_block_duration: int = 15,
    quiet: bool = False,
    logger: logging.Logger = None,
    timetofirst_colname="timetofirst",
    pkts_size_colname="pkts_size",
    pkts_dir_colname="pkts_dir",
    target_colname="app",
    row_id_colname="row_id",
    load_all_columns=False,
) -> Dict[str, FlowpicDataset]:
    #    folder = pathlib.Path(folder)
    #
    #    prefix = f'{dataset_name}_filtered'
    #    if dataset_minpkts != -1:
    #        prefix = f'{prefix}_minpkts{dataset_minpkts}'
    #
    #    fname = folder / 'imc23' / f'{prefix}.parquet'
    #    utils.log_msg(f'loading: {fname}', logger)
    #
    #    if load_all_columns:
    #        df = pd.read_parquet(fname)
    #    else:
    #        df = pd.read_parquet(fname, columns=[timetofirst_colname, pkts_size_colname, pkts_dir_colname, target_colname, row_id_colname])
    #
    #    fname = folder / 'imc23' / f'{prefix}_splits.parquet'
    ##    fname = folder / 'imc23' / f'{prefix}_100samples_splits.parquet'
    #    utils.log_msg(f'loading: {fname}', logger)
    #    df_splits = pd.read_parquet(fname)
    PRESELECTED_COLUMNS = [
        timetofirst_colname,
        pkts_size_colname,
        pkts_dir_colname,
        target_colname,
        row_id_colname,
    ]
    columns = None if load_all_columns else PRESELECTED_COLUMNS

    df = datasets_utils.load_parquet(
        dataset_name, dataset_minpkts, None, columns=columns, animation=True
    )
    path = datasets_utils.get_dataset_parquet_filename(
        dataset_name, dataset_minpkts, None
    )
    utils.log_msg(f"loaded: {path}", logger)

    df_splits = datasets_utils.load_parquet(
        dataset_name, dataset_minpkts, split_idx, animation=True
    )
    path = datasets_utils.get_dataset_parquet_filename(
        dataset_name, dataset_minpkts, split_idx
    )
    utils.log_msg(f"loaded: {path}", logger)

    ser = df_splits[df_splits["split_index"] == split_idx].iloc[0]
    test_indexes = ser["test_indexes"]
    df_test = df.loc[test_indexes]

    params = dict(
        timetofirst_colname=timetofirst_colname,
        pkts_size_colname=pkts_size_colname,
        pkts_dir_colname=pkts_dir_colname,
        target_colname=target_colname,
        flowpic_dim=flowpic_dim,
        flowpic_block_duration=flowpic_block_duration,
        quiet=quiet,
        logger=logger,
    )
    dset = FlowpicDataset(data=df_test, **params)

    if not quiet:
        utils.log_msg(dset.samples_count(), logger)

    name = "test"  # f'kf{inner_kfold}_kf{outer_kfold}'
    dsets = {name: dset}
    return dsets


def generic_load_for_finetuning_dataset(
    folder: pathlib.Path,
    dataset_name: str,
    flowpic_dim: int = 32,
    flowpic_block_duration: int = 15,
    quiet: bool = False,
    logger: logging.Logger = None,
    train_samples: int = 10,
    aug_config=None,
    aug_samples=10,
    aug_when_loading=True,
    aug_yield_also_original=False,
    seed: int = 12345,
    dataset_minpkts: int = -1,
    timetofirst_colname="timetofirst",
    pkts_size_colname="pkts_size",
    pkts_dir_colname="pkts_dir",
    target_colname="app",
    row_id_colname="row_id",
) -> Dict[str, FlowpicDataset]:
    # dset_dict = load_ucdavis_icdm19_test_dataset(
    dset_dict = generic_load_test_dataset(
        folder,
        flowpic_dim=flowpic_dim,
        flowpic_block_duration=flowpic_block_duration,
        quiet=True,
        logger=logger,
        dataset_minpkts=dataset_minpkts,
        dataset_name=dataset_name,
    )

    samples_count = []
    samples_colnames = []
    new_dset_dict = dict()
    params = dict(
        timetofirst_colname=timetofirst_colname,
        pkts_size_colname=pkts_size_colname,
        pkts_dir_colname=pkts_dir_colname,
        target_colname=target_colname,
        flowpic_dim=flowpic_dim,
        flowpic_block_duration=flowpic_block_duration,
        logger=logger,
    )
    for name, dset in dset_dict.items():
        df = dset.df
        df_train, df_test, _ = train_test_split(
            df,
            target_colname=target_colname,
            max_samples_per_class=train_samples,
            seed=seed,
        )
        train_name = f"{name}_train"
        test_name = f"{name}_test"

        # test set is a basic FlowpicDataset
        new_dset_dict[test_name] = FlowpicDataset(data=df_test, **params)

        # when passing augmentation policy, the training
        # dataset is augmented with multi-view
        if aug_config is None:
            new_dset_dict[train_name] = FlowpicDataset(data=df_train, **params)
        else:
            specific_params = dict(
                aug_config=aug_config,
                num_views=aug_samples,
                yield_also_original=aug_yield_also_original,
                seed=seed,
            )
            new_dset_dict[train_name] = MultiViewDataset(
                data=df_train, **params, **specific_params
            )

        samples_count.append(new_dset_dict[train_name].samples_count())
        samples_count.append(new_dset_dict[test_name].samples_count())
        samples_colnames.append(train_name)
        samples_colnames.append(test_name)

    if not quiet:
        df_samples = pd.DataFrame(samples_count, index=samples_colnames).T
        df_samples = df_samples[sorted(df_samples.columns.tolist())]
        utils.log_msg(df_samples, logger)
    return new_dset_dict


def load_mirage22_train_val_datasets(
    # folder,
    split_idx=0,
    flowpic_dim=32,
    flowpic_block_duration=15,
    max_samples_per_class=-1,
    aug_config=None,
    aug_samples=10,
    aug_when_loading=True,
    aug_yield_also_original=False,
    seed=12345,
    n_workers=10,
    quiet=False,
    logger=None,
    train_val_split_ratio: float = 0.8,
    suppress_val_augmentation: bool = False,
    dataset_minpkts: int = -1,
) -> Tuple[FlowpicDataset, FlowpicDataset]:
    return generic_load_train_val_datasets(
        # folder=folder,
        # dataset_name='mirage22',
        dataset_name=tcbench.DATASETS.MIRAGE22,
        dataset_minpkts=dataset_minpkts,
        split_idx=split_idx,
        flowpic_dim=flowpic_dim,
        flowpic_block_duration=flowpic_block_duration,
        max_samples_per_class=max_samples_per_class,
        aug_config=aug_config,
        aug_samples=aug_samples,
        aug_when_loading=aug_when_loading,
        aug_yield_also_original=aug_yield_also_original,
        seed=seed,
        n_workers=n_workers,
        quiet=quiet,
        logger=logger,
        train_val_split_ratio=train_val_split_ratio,
        suppress_val_augmentation=suppress_val_augmentation,
        timetofirst_colname="timetofirst",
        pkts_size_colname="pkts_size",
        target_colname="app",
    )


def load_mirage22_test_dataset(
    # folder:pathlib.Path,
    split_idx: int = 0,
    flowpic_dim: int = 32,
    flowpic_block_duration: int = 15,
    quiet: bool = False,
    logger: logging.Logger = None,
    dataset_minpkts: int = -1,
) -> Dict[str, FlowpicDataset]:
    return generic_load_test_dataset(
        # folder=folder,
        dataset_name=tcbench.DATASETS.MIRAGE22,  #'mirage22',
        dataset_minpkts=dataset_minpkts,
        split_idx=split_idx,
        flowpic_dim=flowpic_dim,
        flowpic_block_duration=flowpic_block_duration,
        quiet=quiet,
        logger=logger,
        timetofirst_colname="timetofirst",
        pkts_size_colname="pkts_size",
        target_colname="app",
    )


def load_mirage22_for_finetuning_dataset(
    folder: pathlib.Path,
    flowpic_dim: int = 32,
    flowpic_block_duration: int = 15,
    quiet: bool = False,
    logger: logging.Logger = None,
    train_samples: int = 10,
    aug_config=None,
    aug_samples=10,
    aug_when_loading=True,
    aug_yield_also_original=False,
    seed: int = 12345,
    dataset_minpkts=-1,
    **kwargs,
):
    return generic_load_for_finetuning_dataset(
        folder=folder,
        dataset_name="mirage22",
        flowpic_dim=flowpic_dim,
        flowpic_block_duration=flowpic_block_duration,
        quiet=quiet,
        logger=logger,
        train_samples=train_samples,
        aug_config=aug_config,
        aug_samples=aug_samples,
        aug_when_loading=aug_when_loading,
        aug_yield_also_original=aug_yield_also_original,
        seed=seed,
        dataset_minpkts=dataset_minpkts,
        timetofirst_colname="timetofirst",
        pkts_size_colname="pkts_size",
        pkts_dir_colname="pkts_dir",
        target_colname="app",
        row_id_colname="row_id",
    )


def load_mirage19_train_val_datasets(
    # folder,
    split_idx=0,
    flowpic_dim=32,
    flowpic_block_duration=15,
    max_samples_per_class=-1,
    aug_config=None,
    aug_samples=10,
    aug_when_loading=True,
    aug_yield_also_original=False,
    seed=12345,
    n_workers=10,
    quiet=False,
    logger=None,
    train_val_split_ratio: float = 0.8,
    suppress_val_augmentation: bool = False,
    dataset_minpkts: int = -1,
) -> Tuple[FlowpicDataset, FlowpicDataset]:
    return generic_load_train_val_datasets(
        # folder=folder,
        # dataset_name='mirage19',
        dataset_name=tcbench.DATASETS.MIRAGE19,
        dataset_minpkts=dataset_minpkts,
        split_idx=split_idx,
        flowpic_dim=flowpic_dim,
        flowpic_block_duration=flowpic_block_duration,
        max_samples_per_class=max_samples_per_class,
        aug_config=aug_config,
        aug_samples=aug_samples,
        aug_when_loading=aug_when_loading,
        aug_yield_also_original=aug_yield_also_original,
        seed=seed,
        n_workers=n_workers,
        quiet=quiet,
        logger=logger,
        train_val_split_ratio=train_val_split_ratio,
        suppress_val_augmentation=suppress_val_augmentation,
        timetofirst_colname="timetofirst",
        pkts_size_colname="pkts_size",
        target_colname="app",
    )


def load_mirage19_test_dataset(
    # folder:pathlib.Path,
    split_idx: int = 0,
    flowpic_dim: int = 32,
    flowpic_block_duration: int = 15,
    quiet: bool = False,
    logger: logging.Logger = None,
    dataset_minpkts: int = -1,
) -> Dict[str, FlowpicDataset]:
    return generic_load_test_dataset(
        # folder=folder,
        dataset_name=tcbench.DATASETS.MIRAGE19,  #'mirage19',
        dataset_minpkts=dataset_minpkts,
        split_idx=split_idx,
        flowpic_dim=flowpic_dim,
        flowpic_block_duration=flowpic_block_duration,
        quiet=quiet,
        logger=logger,
        timetofirst_colname="timetofirst",
        pkts_size_colname="pkts_size",
        target_colname="app",
    )


def load_mirage19_for_finetuning_dataset(
    folder: pathlib.Path,
    flowpic_dim: int = 32,
    flowpic_block_duration: int = 15,
    quiet: bool = False,
    logger: logging.Logger = None,
    train_samples: int = 10,
    aug_config=None,
    aug_samples=10,
    aug_when_loading=True,
    aug_yield_also_original=False,
    seed: int = 12345,
    dataset_minpkts=-1,
    **kwargs,
):
    return generic_load_for_finetuning_dataset(
        folder=folder,
        dataset_name="mirage19",
        flowpic_dim=flowpic_dim,
        flowpic_block_duration=flowpic_block_duration,
        quiet=quiet,
        logger=logger,
        train_samples=train_samples,
        aug_config=aug_config,
        aug_samples=aug_samples,
        aug_when_loading=aug_when_loading,
        aug_yield_also_original=aug_yield_also_original,
        seed=seed,
        dataset_minpkts=dataset_minpkts,
        timetofirst_colname="timetofirst",
        pkts_size_colname="pkts_size",
        pkts_dir_colname="pkts_dir",
        target_colname="app",
        row_id_colname="row_id",
    )


def load_utmobilenet21_for_finetuning_dataset(
    folder: pathlib.Path,
    flowpic_dim: int = 32,
    flowpic_block_duration: int = 15,
    quiet: bool = False,
    logger: logging.Logger = None,
    train_samples: int = 10,
    aug_config=None,
    aug_samples=10,
    aug_when_loading=True,
    aug_yield_also_original=False,
    seed: int = 12345,
    dataset_minpkts=-1,
    **kwargs,
):
    return generic_load_for_finetuning_dataset(
        folder=folder,
        dataset_name="utmobilenet21",
        flowpic_dim=flowpic_dim,
        flowpic_block_duration=flowpic_block_duration,
        quiet=quiet,
        logger=logger,
        train_samples=train_samples,
        aug_config=aug_config,
        aug_samples=aug_samples,
        aug_when_loading=aug_when_loading,
        aug_yield_also_original=aug_yield_also_original,
        seed=seed,
        dataset_minpkts=dataset_minpkts,
        timetofirst_colname="timetofirst",
        pkts_size_colname="pkts_size",
        pkts_dir_colname="pkts_dir",
        target_colname="app",
        row_id_colname="row_id",
    )


def load_dataset(
    dataset_name: str | tcbench.DATASETS,
    dataset_type: MODELING_DATASET_TYPE,
    **kwargs: Dict[str, Any],
) -> FlowpicDataset:
    dataset_name = str(dataset_name).replace("-", "_")
    curr_module = sys.modules[__name__]
    method_name = f"load_{dataset_name}_{dataset_type.value}"
    if not hasattr(curr_module, method_name):
        raise RuntimeError(f"Cannot find a method called {method_name}()")

    method = getattr(curr_module, method_name)
    return method(**kwargs)
