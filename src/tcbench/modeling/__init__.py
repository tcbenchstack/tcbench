from __future__ import annotations
from enum import Enum

from tcbench.core import StringEnum
from tcbench.modeling.columns import (
    COL_PKTS_SIZE,
    COL_PKTS_DIR,
    #COL_PKTS_IAT,
)


class MODELING_DATASET_TYPE(StringEnum):
    """An enumeration to specify which type of dataset to load"""

    TRAIN_VAL = "train_val_datasets"
    TEST = "test_dataset"
    TRAIN_VAL_LEFTOVER = "train_val_leftover_dataset"
    FINETUNING = "for_finetuning_dataset"


class MODELING_INPUT_REPR_TYPE(StringEnum):
    pass
#    FLOWPIC = "flowpic"
#    PKTSERIES = "pktseries"

class MODELING_FEATURE(StringEnum):
    PKTS_SIZE = COL_PKTS_SIZE
    PKTS_DIR = COL_PKTS_DIR
    #PKTS_IAT = COL_PKTS_IAT
    #PKTS_SIZE_TIMES_DIR = "pkts_size_times_dir"

#class MODELING_METHOD_TYPE(StringEnum):
#    MONOLITHIC = "monolithic"
#    XGBOOST = "xgboost"
#    SIMCLR = "simclr"


class MODELING_METHOD_NAME(StringEnum):
    XGBOOST = "ml.xgboost"

from tcbench.modeling.factory import mlmodel_factory
