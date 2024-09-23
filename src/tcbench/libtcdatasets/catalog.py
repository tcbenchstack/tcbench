from __future__ import annotations

import rich.console
import rich.tree as richtree

from collections import UserDict

from tcbench.libtcdatasets import (
    dataset_mirage
)
from tcbench.libtcdatasets.core import Dataset

from tcbench.libtcdatasets.constants import (
    DATASET_NAME,
)

_DATASET_NAME_TO_CLASS = {
    DATASET_NAME.MIRAGE19: dataset_mirage.Mirage19
}

class DatasetsCatalog(UserDict):
    def __init__(
        self,
        #fname_metadata: pathlib.Path = DATASETS_RESOURCES_METADATA_FNAME,
    ):
        super().__init__()
        for dset_name, dset_class in _DATASET_NAME_TO_CLASS.items():
            self.data[str(dset_name)] = dset_class()

    def __getitem__(self, key: Any) -> DatasetMetadata:
        if isinstance(key, DATASET_NAME):
            key = str(key)
        return self.data[str(key)]

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, DATASET_NAME):
            key = str(key)
        return key in self.data

    def __setitem__(self, key: Any, value: Any) -> None:
        raise ValueError(f"{self.__class__.__name__} is immutable")

    def __rich__(self) -> richtree.Tree:
        tree = richtree.Tree("Datasets")
        for dset_name in sorted(self.keys()):
            dset_metadata = self[dset_name]    
            node = richtree.Tree(dset_name)
            node.add(dset_metadata.__rich__())
            tree.add(node)
        return tree

    def __rich_console__(self,
        console: rich.console.Console,
        options: rich.console.ConsoleOptions,
    ) -> rich.console.RenderResult:
        yield self.__rich__()


def datasets_catalog() -> DatasetsCatalog:
    return DatasetsCatalog()

def get_dataset(name: DATASET_NAME) -> Dataset:
    return DatasetsCatalog()[name]
