from __future__ import annotations

from tcbench.libtcdatasets.fileutils import get_module_folder
from tcbench.core import StringEnum

_module_folder = get_module_folder()

DATASETS_RESOURCES_FOLDER = _module_folder / "resources"
DATASETS_RESOURCES_METADATA_FNAME = DATASETS_RESOURCES_FOLDER / "DATASETS_METADATA.yml"
# DATASETS_RESOURCES_YAML_MD5_FNAME = DATASETS_RESOURCES_FOLDER / "DATASETS_FILES_MD5.yml"
DATASETS_DEFAULT_INSTALL_ROOT_FOLDER = _module_folder / "installed_datasets"

APP_LABEL_BACKGROUND = "_background_"
APP_LABEL_ALL = "_all_"


class DATASET_NAME(StringEnum):
    MIRAGE19 = "mirage19"
    MIRAGE22 = "mirage22"
#    UCDAVIS19 = "ucdavis19"
#    UTMOBILENET21 = "utmobilenet21"

class DATASET_TYPE(StringEnum):
    RAW = "raw"
#    PREPROCESS = "preprocess"
    CURATE = "curate"
