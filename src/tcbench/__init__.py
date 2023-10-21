import pathlib

__version__ = "0.0.22"

DEFAULT_AIM_REPO = pathlib.Path("./aim-repo")
DEFAULT_ARTIFACTS_FOLDER = pathlib.Path("./aim-repo/artifacts")

DEFAULT_CAMPAIGN_AUGATLOAD_AUGMENTATIONS = (
    "noaug",
    "rotate",
    "horizontalflip",
    "colorjitter",
    "packetloss",
    "changertt",
    "timeshift",
)
DEFAULT_CAMPAIGN_AUGATLOAD_SEEDS = (12345, 42, 666)
DEFAULT_CAMPAIGN_AUGATLOAD_FLOWPICDIMS = (32, 64, 1500)
DEFAULT_CAMPAIGN_AUGATLOAD_PKTSERIESLEN = (10, 30)

DEFAULT_CAMPAIGN_CONTRALEARNANDFINETUNE_FLOWPICDIMS = (32, 64, 1500)
DEFAULT_CAMPAING_CONTRALEARNANDFINETUNE_SEEDS_CONTRALEARN = (12345, 1, 2, 3, 4)
DEFAULT_CAMPAIGN_CONTRALEARNANDFINETUNE_SEEDS_FINETUNE = (12345, 1, 2, 3, 4)
DEFAULT_CAMPAIGN_CONTRALEARNANDFINETUNE_AUGMENTATIONS = "changertt,timeshift"
DEFAULT_CAMPAIGN_CONTRALEARNANDFINETUNE_VALID_AUGMENTATIONS = tuple([
    aug_name
    for aug_name in DEFAULT_CAMPAIGN_AUGATLOAD_AUGMENTATIONS
    if aug_name != "noaug"
])

from tcbench.libtcdatasets.datasets_utils import (
    get_datasets_root_folder,
    get_dataset_folder,
    DATASETS,
    load_parquet,
)

from tcbench.modeling import (
    MODELING_DATASET_TYPE,
    MODELING_INPUT_REPR_TYPE,
    MODELING_METHOD_TYPE,
)
