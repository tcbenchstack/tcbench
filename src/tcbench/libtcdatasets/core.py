from __future__ import annotations
import rich.progress as richprogress
import polars as pl

# from rich.tree import Tree
# from rich.table import Table
# import rich.box
from typing import Dict, Any
from collections import UserDict, UserList

# import yaml
# import sys
import abc
import pathlib
import dataclasses

# import rich
# import zipfile
# import tempfile
# import requests
# import tarfile
# import enum
# import hashlib

# from tcbench.cli import get_rich_console
# from tcbench.cli.richutils import rich_label, rich_samples_count_report
from tcbench.libtcdatasets.constants import (
    DATASET_NAME,
    DATASETS_DEFAULT_INSTALL_ROOT_FOLDER,
    DATASETS_RESOURCES_METADATA_FNAME,
)
from tcbench.libtcdatasets import fileutils
from tcbench.cli import richutils
from tcbench import _tcbenchrc

# console = get_rich_console()


def get_dataset_folder(dataset_name: str | DATASET_NAME) -> pathlib.Path:
    """Returns the path where a specific datasets in installed"""
    return DATASETS_DEFAULT_INSTALL_ROOT_FOLDER / str(dataset_name)


# def load_datasets_resources_yaml():
#    return fileutils.load_yaml(DATASETS_RESOURCES_YAML_FNAME)


# def load_datasets_files_md5_yaml():
#    return load_yaml(get_module_folder() / "resources" / DATASETS_FILES_MD5_YAML)


@dataclasses.dataclass
class DatasetMetadata:
    num_classes: int = -1
    paper: str = None
    website: str = None
    raw_data_url: str = None
    raw_data_md5: str = None
    curated_data_url: str = None
    curated_data_md5: str = None


class DatasetMetadataCatalog(UserDict):
    def __init__(self):
        super().__init__()
        for dataset_name, dataset_metadata in fileutils.load_yaml(
            DATASETS_RESOURCES_METADATA_FNAME
        ).items():
            self.data[dataset_name] = DatasetMetadata(**dataset_metadata)

    def __getitem__(self, key: Any) -> DatasetMetadata:
        if isinstance(key, DATASET_NAME):
            key = str(key)
        return self.data[str(key)]

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, DATASET_NAME):
            key = str(key)
        return key in self.data

    def __setitem__(self, key: Any, value: Any) -> None:
        raise ValueError(f"DatasetMetadataCatalog is immutable")


class RawDatasetInstaller:
    def __init__(
        self,
        url: str,
        install_folder: pathlib.Path = None,
        verify_tls: bool = True,
        force_reinstall: bool = False,
    ):
        self.url = url
        self.install_folder = install_folder
        self.verify_tls = verify_tls
        self.force_reinstall = force_reinstall
        self.download_path = None

        if install_folder is None:
            self.install_folder = DATASETS_DEFAULT_INSTALL_ROOT_FOLDER

        self.install()

    def install(self):
        self.download_path = self.download()
        self.unpack(self.download_path)

    def download(self) -> pathlib.Path:
        return fileutils.download_url(
            self.url,
            self.install_folder / "download",
            self.verify_tls,
            self.force_reinstall,
        )

    def unpack(self, path: pathlib.Path) -> pathlib.Path:
        func_unpack = None
        if path.suffix == ".zip":
            func_unpack = fileutils.unzip
        elif str(path).endswith(".tar.gz"):
            func_unpack = fileutils.untar
        else:
            raise RuntimeError(f"Unrecognized {path.suffix} archive")

        dst = self.install_folder / "raw"
        if self.force_reinstall or not dst.exists() or len(list(dst.iterdir())) == 0:
            return func_unpack(src=path, dst=dst)
        return dst


class RawDataset:
    def __init__(self, name: DATASET_NAME):
        self.name = name
        self.metadata = DatasetMetadataCatalog()[name]
        self.install_folder = _tcbenchrc.install_folder / str(self.name)
        self.df = None
        self.df_stats = None

    @property
    def folder_download(self):
        return self.install_folder / "download"

    @property
    def folder_raw(self):
        return self.install_folder / "raw"

    @property
    def folder_preprocess(self):
        return self.install_folder / "preprocess"

    @property
    def folder_curate(self):
        return self.install_folder / "curate"

    def install(self) -> pathlib.Path:
        self._raw_install()
        self.preprocess()
        self.curate()
        return self.install_folder

    def _raw_install(self) -> pathlib.Path:
        RawDatasetInstaller(
            url=self.metadata.raw_data_url,
            install_folder=self.install_folder,
            verify_tls=True,
            force_reinstall=True,
        )
        return self.install_folder

    def preprocess(self) -> None:
        pass

    def curate(self) -> None:
        pass

    @abc.abstractmethod
    def load(self, *args, **kwargs) -> pl.DataFrame:
        pass


class SequentialPipeStage:
    def __init__(self, func, name:str = None, **kwargs):
        self.func = func
        self.name = name if name else ""
        self.run_kwargs = kwargs

    def run(self, data:Any) -> Any:
        return self.func(data, **self.run_kwargs)


class SequentialPipe(UserList):
    def __init__(
        self, 
        *stages: SequentialPipelineStage,
        name: str = None, 
        progress: bool = True
    ):
        super().__init__(stages)
        self.name = name if name is not None else ""
        self.progress = progress

    def run(self, data:Any) -> Any:
        with richutils.SpinnerAndCounterProgress(
            description=self.name,
            total=len(self.data),
            visible=self.progress
        ) as progress:
            for idx, stage in enumerate(self.data):
                progress.update_description(
                    " ".join([self.name, stage.name]).strip()
                )
                data = stage.run(data)
                progress.update()
        return data 
        

# def install_ucdavis_icdm19(input_folder, num_workers=10, *args, **kwargs):
#    # moved here to speedup loading
#    from tcbench.libtcdatasets import ucdavis_icdm19_csv_to_parquet
#    from tcbench.libtcdatasets import ucdavis_icdm19_generate_splits
#
#    rich_label("unpack")
#    expected_files = [
#        "pretraining.zip",
#        "Retraining(human-triggered).zip",
#        "Retraining(script-triggered).zip",
#    ]
#
#    input_folder = pathlib.Path(input_folder)
#    for fname in expected_files:
#        path = input_folder / fname
#        if not path.exists():
#            raise RuntimeError(f"missing {path}")
#
#    dataset_folder = get_dataset_folder(DATASETS.UCDAVISICDM19)
#
#    # unpack the raw CSVs
#    raw_folder = dataset_folder / "raw"
#    if not raw_folder.exists():
#        raw_folder.mkdir(parents=True)
#    for fname in expected_files:
#        path = input_folder / fname
#        unzip(path, raw_folder)
#
#    # preprocess raw CSVs
#    rich_label("preprocess", extra_new_line=True)
#    preprocessed_folder = dataset_folder / "preprocessed"
#    cmd = f"--input-folder {raw_folder} --num-workers {num_workers} --output-folder {preprocessed_folder}"
#    args = ucdavis_icdm19_csv_to_parquet.cli_parser().parse_args(cmd.split())
#    ucdavis_icdm19_csv_to_parquet.main(args)
#
#    # generate data splits
#    rich_label("generate splits", extra_new_line=True)
#    ucdavis_icdm19_generate_splits.main(
#        dict(datasets={str(DATASETS.UCDAVISICDM19): preprocessed_folder})
#    )
#
#    verify_dataset_md5s(DATASETS.UCDAVISICDM19)
#
#
# def install_utmobilenet21(input_folder, num_workers=50):
#    # moved here to speed up loading
#    from tcbench.libtcdatasets import utmobilenet21_csv_to_parquet
#    from tcbench.libtcdatasets import utmobilenet21_generate_splits
#
#    # enforcing this to 50, attempting to replicate the
#    # original setting used to create the artifact
#    num_workers=50
#
#    rich_label("unpack")
#    expected_files = ["UTMobileNet2021.zip"]
#    input_folder = pathlib.Path(input_folder)
#
#    for fname in expected_files:
#        path = input_folder / fname
#        if not path.exists():
#            raise RuntimeError(f"missing {path}")
#    # dataset_folder = _get_module_folder() / FOLDER_DATASETS / 'utmobilenet21'
#    dataset_folder = get_dataset_folder(DATASETS.UTMOBILENET21)
#
#    # unpack the raw CSVs
#    raw_folder = dataset_folder / "raw"
#    if not raw_folder.exists():
#        raw_folder.mkdir(parents=True)
#    for fname in expected_files:
#        path = input_folder / fname
#        unzip(path, raw_folder)
#
#    # preprocess raw CSVs
#    rich_label("preprocess", extra_new_line=True)
#    preprocessed_folder = dataset_folder / "preprocessed"
#    cmd = f"--input-folder {raw_folder} --num-workers {num_workers} --output-folder {preprocessed_folder}"
#    args = utmobilenet21_csv_to_parquet.cli_parser().parse_args(cmd.split())
#    utmobilenet21_csv_to_parquet.main(args)
#
#    # generate data splits
#    rich_label("filter & generate splits", extra_new_line=True)
#    cmd = f"--config dummy_file.txt"
#    args = utmobilenet21_generate_splits.cli_parser().parse_args(cmd.split())
#    args.config = dict(datasets={str(DATASETS.UTMOBILENET21): preprocessed_folder})
#    utmobilenet21_generate_splits.main(args)
#
#    #verify_dataset_md5s(DATASETS.UTMOBILENET21)
#
#
# def install_mirage22(input_folder=None, num_workers=30):
#    # moved here to speed up loading
#    from tcbench.libtcdatasets import mirage22_json_to_parquet
#    from tcbench.libtcdatasets import mirage22_generate_splits
#
#    rich_label("download & unpack")
#    expected_files = ["MIRAGE-COVID-CCMA-2022.zip"]
#    # dataset_folder = _get_module_folder() / FOLDER_DATASETS / 'mirage22'
#    dataset_folder = get_dataset_folder(DATASETS.MIRAGE22)
#
#    raw_folder = dataset_folder / "raw"
#    if input_folder:
#        _verify_expected_files_exists(input_folder, expected_files)
#        unzip(input_folder / expected_files[0], raw_folder)
#    else:
#        datasets_yaml = load_datasets_yaml()
#        url = datasets_yaml[str(DATASETS.MIRAGE22)]["data"]
#        with tempfile.TemporaryDirectory() as tmpfolder:
#            path = download_url(url, tmpfolder)
#            unzip(path, raw_folder)
#
#    # second unzip
#    files = [
#        "Discord.zip",
#        "GotoMeeting.zip",
#        "Meet.zip",
#        "Messenger.zip",
#        "Skype.zip",
#        "Slack.zip",
#        "Teams.zip",
#        "Webex.zip",
#        "Zoom.zip",
#    ]
#    raw_folder = raw_folder / "MIRAGE-COVID-CCMA-2022" / "Raw_JSON"
#    for fname in files:
#        path = raw_folder / fname
#        unzip(path, raw_folder)
#
#    rich_label("preprocess", extra_new_line=True)
#    preprocess_folder = dataset_folder / "preprocessed"
#    cmd = f"--input-folder {raw_folder} --output-folder {preprocess_folder}"
#    args = mirage22_json_to_parquet.cli_parser().parse_args(cmd.split())
#    mirage22_json_to_parquet.main(args)
#
#    rich_label("filter & generate splits", extra_new_line=True)
#    # fooling the script believe there is a config.yml
#    cmd = f"--config dummy_file.txt"
#    args = mirage22_generate_splits.cli_parser().parse_args(cmd.split())
#    args.config = dict(datasets={str(DATASETS.MIRAGE22): preprocess_folder})
#    mirage22_generate_splits.main(args)
#
#    verify_dataset_md5s(DATASETS.MIRAGE22)
#
#
# def install_mirage19(input_folder=None, num_workers=30):
#    from tcbench.libtcdatasets import mirage19_json_to_parquet
#    from tcbench.libtcdatasets import mirage19_generate_splits
#
#    rich_label("download & unpack")
#    expected_files = [
#        "MIRAGE-2019_traffic_dataset_downloadable_v2.tar.gz",
#    ]
#    # dataset_folder = _get_module_folder() / FOLDER_DATASETS / 'mirage19'
#    dataset_folder = get_dataset_folder(DATASETS.MIRAGE19)
#
#    raw_folder = dataset_folder / "raw"
#    if input_folder:
#        _verify_expected_files_exists(input_folder, expected_files)
#        untar(input_folder / expected_files[0], raw_folder)
#    else:
#        datasets_yaml = load_datasets_yaml()
#        url = datasets_yaml[str(DATASETS.MIRAGE19)]["data"]
#        with tempfile.TemporaryDirectory() as tmpfolder:
#            path = download_url(url, tmpfolder)
#            untar(path, raw_folder)
#
#    rich_label("preprocess", extra_new_line=True)
#    preprocess_folder = dataset_folder / "preprocessed"
#    cmd = f"--input-folder {raw_folder} --output-folder {preprocess_folder} --num-workers {num_workers}"
#    args = mirage19_json_to_parquet.cli_parser().parse_args(cmd.split())
#    mirage19_json_to_parquet.main(
#        args.input_folder, args.output_folder / "mirage19.parquet", args.num_workers
#    )
#
#    rich_label("filter & generate splits", extra_new_line=True)
#    # fooling the script believe there is a config.yml
#    cmd = f"--config dummy_file.txt"
#    args = mirage19_generate_splits.cli_parser().parse_args(cmd.split())
#    args.config = dict(datasets={str(DATASETS.MIRAGE19): preprocess_folder})
#    mirage19_generate_splits.main(args)
#
#    verify_dataset_md5s(DATASETS.MIRAGE19)
#
#
# def install(dataset_name, *args, **kwargs):
#    dataset_name = str(dataset_name).replace("-", "_")
#    curr_module = sys.modules[__name__]
#    func_name = f"install_{dataset_name}"
#    func = getattr(curr_module, func_name)
#    return func(*args, **kwargs)
#
#
# def get_dataset_parquet_filename(
#    dataset_name: str | DATASETS, min_pkts: int = -1, split: str = None, animation=False
# ) -> pathlib.Path:
#    """Returns the path of a dataset parquet file
#
#    Arguments:
#        dataset_name: The name of the dataset
#        min_pkts: the filtering rule applied when curating the datasets.
#            If -1, load the unfiltered dataset
#        split: if min_pkts!=-1, is used to request the loading of
#            the split file. For DATASETS.UCDAVISICDM19
#            values can be "human", "script" or a number
#            between 0 and 4.
#            For all other dataset split can be anything
#            which is not None (e.g., True)
#
#    Returns:
#        The pathlib.Path of a dataset parquet file
#    """
#    dataset_folder = get_dataset_folder(dataset_name) / "preprocessed"
#    path = dataset_folder / f"{dataset_name}.parquet"
#
#    if isinstance(split, int) and split < 0:
#        split = None
#
#    #    if isinstance(split, bool):
#    #        split = 0
#    #    elif isinstance(split, int):
#    #        split = str(split)
#    #
#    #    if min_pkts == -1 and (split is None or int(split) < 0):
#    #        return path
#    #
#    if isinstance(dataset_name, str):
#        dataset_name = DATASETS.from_str(dataset_name)
#
#    if dataset_name != DATASETS.UCDAVISICDM19:
#        if min_pkts != -1:
#            dataset_folder /= "imc23"
#            if split is None:
#                path = (
#                    dataset_folder
#                    / f"{dataset_name}_filtered_minpkts{min_pkts}.parquet"
#                )
#            else:
#                path = (
#                    dataset_folder
#                    / f"{dataset_name}_filtered_minpkts{min_pkts}_splits.parquet"
#                )
#    else:
#        #        if split is None:
#        #            raise RuntimeError('split cannot be None for ucdavis-icdm19')
#        #        dataset_folder /= 'imc23'
#        #        if split in ('human', 'script'):
#        #            path = dataset_folder / f'test_split_{split}.parquet'
#        #        else:
#        #            if split == 'train':
#        #                split = 0
#        #            path = dataset_folder / f'train_split_{split}.parquet'
#
#        if split is not None:
#            dataset_folder /= "imc23"
#            if split in ("human", "script"):
#                path = dataset_folder / f"test_split_{split}.parquet"
#            else:
#                if split == "train":
#                    split = 0
#                path = dataset_folder / f"train_split_{split}.parquet"
#
#    return path
#
#
# def load_parquet(
#    dataset_name: str | DATASETS,
#    min_pkts: int = -1,
#    split: str = None,
#    columns: List[str] = None,
#    animation: bool = False,
# ) -> pd.DataFrame:
#    """Load and returns a dataset parquet file
#
#    Arguments:
#        dataset_name: The name of the dataset
#        min_pkts: the filtering rule applied when curating the datasets.
#            If -1, load the unfiltered dataset
#        split: if min_pkts!=-1, is used to request the loading of
#            the split file. For DATASETS.UCDAVISICDM19
#            values can be "human", "script" or a number
#            between 0 and 4.
#            For all other dataset split can be anything
#            which is not None (e.g., True)
#        columns: A list of columns to load (if None, load all columns)
#        animation: if True, create a loading animation on the console
#
#    Returns:
#        A pandas dataframe and the related parquet file used to load the dataframe
#    """
#    path = get_dataset_parquet_filename(dataset_name, min_pkts, split)
#
#    import pandas as pd
#    from tcbench import cli
#
#    if animation:
#        with cli.console.status(f"loading: {path}...", spinner="dots"):
#            return pd.read_parquet(path, columns=columns)
#    return pd.read_parquet(path, columns=columns)
#
#
# def get_split_indexes(dataset_name, min_pkts=-1):
#    dataset_path = get_dataset_folder(dataset_name) / "preprocessed" / "imc23"
#    if str(dataset_name) == str(DATASETS.UCDAVISICDM19):  #'ucdavis-icdm19':
#        # automatically detect all split indexes
#        split_indexes = sorted(
#            [
#                int(split_path.stem.rsplit("_", 1)[1])
#                for split_path in dataset_path.glob("train_split*")
#            ]
#        )
#
#    # elif args.dataset in (str(DATASETS.MIRAGE19), str(DATASETS.MIRAGE22), str(DATASETS.UTMOBILENET21)): #'mirage19', 'mirage22', 'utmobilenet21'):
#    else:
#        #        prefix = f'{args.dataset}_filtered'
#        #        if args.dataset_minpkts != -1:
#        #            prefix = f'{prefix}_minpkts{args.dataset_minpkts}'
#        #        df_splits = pd.read_parquet(dataset_path / f'{prefix}_splits.parquet')
#        #        split_indexes = df_splits['split_index'].unique().tolist()
#
#        #    else:
#        #        df_splits = pd.read_parquet(dataset_path / f'{args.dataset}_filtered_splits.parquet')
#        #        df_splits = df_splits[df_splits['idx_inner_kfold'] == 0]
#        #        split_indexes = list(map(int, df_splits['split_index'].values))
#        df_splits = load_parquet(dataset_name, min_pkts=min_pkts, split=True)
#        split_indexes = list(map(int, df_splits["split_index"].values))
#
#    #    if args.max_train_splits == -1:
#    #        args.max_train_splits = len(split_indexes)
#    #
#    #    split_indexes = split_indexes[:min(len(split_indexes), args.max_train_splits)]
#
#    return split_indexes
#
# def import_dataset(dataset_name, path_archive):
#    data = load_datasets_yaml()
#    folder_datasets = _get_module_folder() #/ FOLDER_DATASETS
#
#    if dataset_name is None or str(dataset_name) not in data:
#        raise RuntimeError(f"Invalid dataset name {dataset_name}")
#
#    with tempfile.TemporaryDirectory() as tmpfolder:
#        if path_archive is None:
#            dataset_name = str(dataset_name)
#            if "data_curated" not in data[dataset_name]:
#                raise RuntimeError(f"The curated dataset cannot be downloaded (likely for licencing problems). Regenerate it using `tcbench datasets install --name {dataset_name}`")
#            url = data[dataset_name]["data_curated"]
#            expected_md5 = data[dataset_name]["data_curated_md5"]
#
#            try:
#                path_archive = download_url(url, tmpfolder)
#            except requests.exceptions.SSLError:
#                path_archive = download_url(url, tmpfolder, verify=False)
#
#            md5 = get_md5(path_archive)
#            assert md5 == expected_md5, f"MD5 check error: found {md5} while should be {expected_md5}"
#        untar(path_archive, folder_datasets)
#
# def verify_dataset_md5s(dataset_name):
#
#    def flatten_dict(data):
#        res = []
#        for key, value in data.items():
#            key = pathlib.Path(key)
#            if isinstance(value, str):
#                res.append((key, value))
#                continue
#            for inner_key, inner_value in flatten_dict(value):
#                res.append((key / inner_key, inner_value))
#        return res
#
#    dataset_name = str(dataset_name)
#    data_md5 = load_datasets_files_md5_yaml().get(dataset_name, None)
#    expected_files = flatten_dict(data_md5)
#
#    if dataset_name in (None, "") or data_md5 is None:
#        raise RuntimeError(f"Invalid dataset name {dataset_name}")
#
#    folder_dataset = _get_module_folder() / FOLDER_DATASETS / dataset_name
#    if not folder_dataset.exists():
#        raise RuntimeError(f"Dataset {dataset_name} is not installed. Run first \"tcbench datasets install --name {dataset_name}\"")
#
#    folder_dataset /= "preprocessed"
#
#    mismatches = dict()
#    for exp_path, exp_md5 in richprogress.track(expected_files, description="Verifying parquet MD5..."):
#        path = folder_dataset / exp_path
#        if not path.exists():
#            raise RuntimeError(f"File {path} not found")
#
#        found_md5 = get_md5(path)
#        fname = path.name
#        if found_md5 == exp_md5:
#            continue
#        mismatches[path] = (exp_md5, found_md5)
#
#    if mismatches:
#        console.print(f"Found {len(mismatches)}/{len(expected_files)} mismatches when verifying parquet files md5")
#        for path, (expected_md5, found_md5) in mismatches.items():
#            console.print()
#            console.print(f"path: {path}")
#            console.print(f"expected_md5: {expected_md5}")
#            console.print(f"found_md5: {found_md5}")
#    else:
#        console.print("All MD5 are correct!")
#
