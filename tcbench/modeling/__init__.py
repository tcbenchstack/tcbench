from enum import Enum


class MODELING_DATASET_TYPE(Enum):
    """An enumeration to specify which type of dataset to load"""

    TRAIN_VAL = "train_val_datasets"
    TEST = "test_dataset"
    TRAIN_VAL_LEFTOVER = "train_val_leftover_dataset"
    FINETUNING = "for_finetuning_dataset"


class MODELING_INPUT_REPR_TYPE(Enum):
    FLOWPIC = "flowpic"
    PKTSERIES = "pktseries"

    @classmethod
    def from_str(cls, text):
        for member in cls.__members__.values():
            if member.value == text:
                return member
        return None

    def __str__(self):
        return self.value


class MODELING_METHOD_TYPE(Enum):
    MONOLITHIC = "monolithic"
    XGBOOST = "xgboost"
    SIMCLR = "simclr"

    @classmethod
    def from_str(cls, text):
        for member in cls.__members__.values():
            if member.value == text:
                return member
        return None

    def __str__(self):
        return self.value
