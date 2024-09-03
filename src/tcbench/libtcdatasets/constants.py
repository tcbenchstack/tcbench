import enum

from tcbench.libtcdatasets.fileutils import get_module_folder

_module_folder = get_module_folder()

DATASETS_RESOURCES_FOLDER = _module_folder / "resources"
DATASETS_RESOURCES_METADATA_FNAME = DATASETS_RESOURCES_FOLDER / "METADATA.yml"
#DATASETS_RESOURCES_YAML_MD5_FNAME = DATASETS_RESOURCES_FOLDER / "DATASETS_FILES_MD5.yml"
DATASETS_DEFAULT_INSTALL_ROOT_FOLDER = _module_folder / "installed_datasets"

class DATASET_NAME(enum.Enum):
    UCDAVIS19 = "ucdavis19"
    UTMOBILENET21 = "utmobilenet21"
    MIRAGE19 = "mirage19"
    MIRAGE22 = "mirage22"

    @classmethod
    def from_str(cls, text):
        for member in cls.__members__.values():
            if member.value == text:
                return member
        return None

    def __str__(self):
        return self.value
