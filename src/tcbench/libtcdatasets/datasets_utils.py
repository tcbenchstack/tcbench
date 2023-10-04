from __future__ import annotations
import rich.progress as richprogress

from rich.tree import Tree
from rich.table import Table
import rich.box
from typing import Dict, Any

import yaml
import sys
import pathlib
import rich
import zipfile
import tempfile
import requests
import tarfile
import enum
import hashlib

from tcbench.cli import get_rich_console
from tcbench.cli.richutils import rich_label, rich_samples_count_report

FNAME_DATASET_YAML = "DATASETS.yml"
DATASETS_FILES_MD5_YAML = "DATASETS_FILES_MD5.yml"
FOLDER_DATASETS = "datasets"

console = get_rich_console()


class DATASETS(enum.Enum):
    UCDAVISICDM19 = "ucdavis-icdm19"
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


def load_yaml(fname: pathlib.Path) -> Dict[Any, Any]:
    """Load an input YAML file

    Arguments:
        fname: the YAML filename to load

    Return:
        The YAML object loaded
    """
    with open(fname) as fin:
        return yaml.safe_load(fin)


def load_config(fname: pathlib.Path) -> Dict:
    """Load the configuration file of the framework

    Arguments:
        fname: the YAML config file to load

    Return:
        The loaded config file
    """
    return load_yaml(fname)

def get_md5(path: pathlib.Path) -> str:
    h = hashlib.new('md5')
    with open(str(path), "rb") as fin:
        h.update(fin.read())
    return h.hexdigest()

def _get_module_folder():
    curr_module = sys.modules[__name__]
    folder = pathlib.Path(curr_module.__file__).parent
    return folder


def load_datasets_yaml():
    fname = _get_module_folder() / "resources" / FNAME_DATASET_YAML
    with open(fname) as fin:
        data = yaml.safe_load(fin)
    return data


def load_datasets_files_md5_yaml():
    return load_yaml( _get_module_folder() / "resources" / DATASETS_FILES_MD5_YAML)


def get_rich_tree_datasets_properties(dataset_name=None):
    data = load_datasets_yaml()
    folder_datasets = _get_module_folder() / FOLDER_DATASETS

    if dataset_name:
        dataset_name = str(dataset_name)

    tree = Tree("Datasets")
    for curr_dataset_name, attributes in data.items():
        if dataset_name is not None and curr_dataset_name != dataset_name:
            continue

        node = tree.add(curr_dataset_name)

        curr_dataset_folder = folder_datasets / curr_dataset_name
        is_installed = (curr_dataset_folder / "raw").exists()
        is_preprocessed = (
            curr_dataset_folder / "preprocessed" / f"{curr_dataset_name}.parquet"
        ).exists()
        has_splits = (curr_dataset_folder / "preprocessed" / "imc23").exists()

        table = Table(show_header=False, box=None)
        table.add_column("property")
        table.add_column("value", overflow="fold")

        table.add_row(":triangular_flag: classes:", str(attributes["num_classes"]))
        table.add_row(":link: paper_url:", attributes["paper"])
        table.add_row(":link: website:", attributes["website"])
        table.add_row(":link: data:", attributes["data"])
        if "data_curated" in attributes:
            table.add_row(":link: curated data:", attributes["data_curated"])
            table.add_row(":heavy_plus_sign: curated data MD5:", attributes["data_curated_md5"])

        if is_installed:
            path = curr_dataset_folder / "raw"
            text = f"[green]{path}[/green]"
        else:
            text = "[red]None[/red]"
        table.add_row(":file_folder: installed:", text)

        if is_preprocessed:
            path = curr_dataset_folder / "preprocessed"
            text = f"[green]{path}[/green]"
        else:
            text = "[red]None[/red]"
        table.add_row(":file_folder: preprocessed:", text)

        if has_splits:
            path = curr_dataset_folder / "preprocessed" / "imc23"
            text = f"[green]{path}[/green]"
        else:
            text = f"[red]None[/red]"
        table.add_row(":file_folder: data splits:", text)

        node.add(table)

    return tree


def get_rich_tree_parquet_files(dataset_name=None):
    data = load_datasets_yaml()
    folder_datasets = _get_module_folder() / FOLDER_DATASETS

    if dataset_name:
        dataset_name = str(dataset_name)

    tree = Tree("Datasets")
    for curr_dataset_name, attributes in data.items():
        if dataset_name is not None and curr_dataset_name != dataset_name:
            continue

        node = tree.add(curr_dataset_name)

        preprocessed = Tree(":file_folder: preprocessed/")
        preprocessed.add(f"{curr_dataset_name}.parquet")
        
        path = folder_datasets / curr_dataset_name / "preprocessed" / "LICENSE"
        if path.exists():
            preprocessed.add("LICENSE")

        imc23 = Tree(":file_folder: imc23/")
        folder = folder_datasets / curr_dataset_name / "preprocessed" / "imc23"
        for path in sorted(folder.glob("*.parquet")):
            imc23.add(path.name)
        preprocessed.add(imc23)

        node.add(preprocessed)
    return tree


def get_rich_dataset_schema(dataset_name, schema_type):
    folder = get_dataset_resources_folder()
    path = folder / f"{dataset_name}.yml"
    data = load_yaml(path)
    if dataset_name == DATASETS.UCDAVISICDM19:
        schema_type = "__all__"
    else:
        schema_type = f"__{schema_type}__"
    schema = data[schema_type]
    table = Table()
    table.add_column("Field")
    table.add_column("Dtype")
    table.add_column("Description", overflow="fold")
    for name, attrs in schema.items():
        table.add_row(name, attrs["dtype"], attrs["description"])
    return table


def download_url(url: str, save_to: pathlib.Path, verify:bool =True) -> pathlib.Path:
    """Download a dataset tarball.

    Args:
        url: the object URL
        save_to: an optional destination folder. If dst is None, the tarball is placed at
            the root_folder specified by the archive (or downloadutils.DEFAULT_DOWNLOAD_FOLDER
            when no root_folder is specified). Note that if dst != root_folder,
            the internal metadata will be adjusted using install()

    Returns:
        the path of the downloaded file
    """
    save_to = pathlib.Path(save_to)

    fname = pathlib.Path(url).name
    save_as = save_to / fname

    resp = requests.get(url, stream=True, verify=verify)
    totalbytes = int(resp.headers.get("content-length", 0))

    if not save_as.parent.exists():
        save_as.parent.mkdir(parents=True)

    with open(str(save_as), "wb") as fout, richprogress.Progress(
        richprogress.TextColumn("[progress.description]{task.description}"),
        richprogress.BarColumn(),
        richprogress.FileSizeColumn(),
        richprogress.TextColumn("/"),
        richprogress.TotalFileSizeColumn(),
        richprogress.TextColumn("eta"),
        richprogress.TimeRemainingColumn(),
        console=console,
    ) as progressbar:
        task_id = progressbar.add_task("Downloading...", total=totalbytes)
        for data in resp.iter_content(chunk_size=64 * 1024):
            size = fout.write(data)
            progressbar.advance(task_id, advance=size)

    return save_as


def _verify_expected_files_exists(folder, expected_files):
    for fname in expected_files:
        path = folder / fname
        if not path.exists():
            raise RuntimeError(f"missing {path}")


def unzip(src: pathlib.Path, dst: pathlib.Path):
    if not dst.exists():
        dst.mkdir(parents=True)
    print(f"opening: {src}")
    with zipfile.ZipFile(src) as fzipped:
        fzipped.extractall(dst)


def untar(src: pathlib.Path, dst: pathlib.Path):
    if not dst.exists():
        dst.mkdir(parents=True)
    print(f"opening: {src}")
    ftar = tarfile.open(src, "r:gz")
    ftar.extractall(dst)
    ftar.close()


def get_datasets_root_folder() -> pathlib.Path:
    """Returns the path where datasets all datasets are installed"""
    return _get_module_folder() / FOLDER_DATASETS


def get_dataset_folder(dataset_name: str | DATASETS) -> pathlib.Path:
    """Returns the path where a specific datasets in installed"""
    if dataset_name:
        dataset_name = str(dataset_name)
    return _get_module_folder() / FOLDER_DATASETS / dataset_name


def get_dataset_resources_folder():
    return _get_module_folder() / "resources"


def install_ucdavis_icdm19(input_folder, num_workers=10, *args, **kwargs):
    # moved here to speedup loading
    from tcbench.libtcdatasets import ucdavis_icdm19_csv_to_parquet
    from tcbench.libtcdatasets import ucdavis_icdm19_generate_splits

    rich_label("unpack")
    expected_files = [
        "pretraining.zip",
        "Retraining(human-triggered).zip",
        "Retraining(script-triggered).zip",
    ]

    input_folder = pathlib.Path(input_folder)
    for fname in expected_files:
        path = input_folder / fname
        if not path.exists():
            raise RuntimeError(f"missing {path}")

    dataset_folder = get_dataset_folder(DATASETS.UCDAVISICDM19)

    # unpack the raw CSVs
    raw_folder = dataset_folder / "raw"
    if not raw_folder.exists():
        raw_folder.mkdir(parents=True)
    for fname in expected_files:
        path = input_folder / fname
        unzip(path, raw_folder)

    # preprocess raw CSVs
    rich_label("preprocess", extra_new_line=True)
    preprocessed_folder = dataset_folder / "preprocessed"
    cmd = f"--input-folder {raw_folder} --num-workers {num_workers} --output-folder {preprocessed_folder}"
    args = ucdavis_icdm19_csv_to_parquet.cli_parser().parse_args(cmd.split())
    ucdavis_icdm19_csv_to_parquet.main(args)

    # generate data splits
    rich_label("generate splits", extra_new_line=True)
    ucdavis_icdm19_generate_splits.main(
        dict(datasets={str(DATASETS.UCDAVISICDM19): preprocessed_folder})
    )

    verify_dataset_md5s(DATASETS.UCDAVISICDM19)


def install_utmobilenet21(input_folder, num_workers=50):
    # moved here to speed up loading
    from tcbench.libtcdatasets import utmobilenet21_csv_to_parquet
    from tcbench.libtcdatasets import utmobilenet21_generate_splits

    # enforcing this to 50, attempting to replicate the 
    # original setting used to create the artifact
    num_workers=50

    rich_label("unpack")
    expected_files = ["UTMobileNet2021.zip"]
    input_folder = pathlib.Path(input_folder)

    for fname in expected_files:
        path = input_folder / fname
        if not path.exists():
            raise RuntimeError(f"missing {path}")
    # dataset_folder = _get_module_folder() / FOLDER_DATASETS / 'utmobilenet21'
    dataset_folder = get_dataset_folder(DATASETS.UTMOBILENET21)

    # unpack the raw CSVs
    raw_folder = dataset_folder / "raw"
    if not raw_folder.exists():
        raw_folder.mkdir(parents=True)
    for fname in expected_files:
        path = input_folder / fname
        unzip(path, raw_folder)

    # preprocess raw CSVs
    rich_label("preprocess", extra_new_line=True)
    preprocessed_folder = dataset_folder / "preprocessed"
    cmd = f"--input-folder {raw_folder} --num-workers {num_workers} --output-folder {preprocessed_folder}"
    args = utmobilenet21_csv_to_parquet.cli_parser().parse_args(cmd.split())
    utmobilenet21_csv_to_parquet.main(args)

    # generate data splits
    rich_label("filter & generate splits", extra_new_line=True)
    cmd = f"--config dummy_file.txt"
    args = utmobilenet21_generate_splits.cli_parser().parse_args(cmd.split())
    args.config = dict(datasets={str(DATASETS.UTMOBILENET21): preprocessed_folder})
    utmobilenet21_generate_splits.main(args)

    #verify_dataset_md5s(DATASETS.UTMOBILENET21)


def install_mirage22(input_folder=None, num_workers=30):
    # moved here to speed up loading
    from tcbench.libtcdatasets import mirage22_json_to_parquet
    from tcbench.libtcdatasets import mirage22_generate_splits

    rich_label("download & unpack")
    expected_files = ["MIRAGE-COVID-CCMA-2022.zip"]
    # dataset_folder = _get_module_folder() / FOLDER_DATASETS / 'mirage22'
    dataset_folder = get_dataset_folder(DATASETS.MIRAGE22)

    raw_folder = dataset_folder / "raw"
    if input_folder:
        _verify_expected_files_exists(input_folder, expected_files)
        unzip(input_folder / expected_files[0], raw_folder)
    else:
        datasets_yaml = load_datasets_yaml()
        url = datasets_yaml[str(DATASETS.MIRAGE22)]["data"]
        with tempfile.TemporaryDirectory() as tmpfolder:
            path = download_url(url, tmpfolder)
            unzip(path, raw_folder)

    # second unzip
    files = [
        "Discord.zip",
        "GotoMeeting.zip",
        "Meet.zip",
        "Messenger.zip",
        "Skype.zip",
        "Slack.zip",
        "Teams.zip",
        "Webex.zip",
        "Zoom.zip",
    ]
    raw_folder = raw_folder / "MIRAGE-COVID-CCMA-2022" / "Raw_JSON"
    for fname in files:
        path = raw_folder / fname
        unzip(path, raw_folder)

    rich_label("preprocess", extra_new_line=True)
    preprocess_folder = dataset_folder / "preprocessed"
    cmd = f"--input-folder {raw_folder} --output-folder {preprocess_folder}"
    args = mirage22_json_to_parquet.cli_parser().parse_args(cmd.split())
    mirage22_json_to_parquet.main(args)

    rich_label("filter & generate splits", extra_new_line=True)
    # fooling the script believe there is a config.yml
    cmd = f"--config dummy_file.txt"
    args = mirage22_generate_splits.cli_parser().parse_args(cmd.split())
    args.config = dict(datasets={str(DATASETS.MIRAGE22): preprocess_folder})
    mirage22_generate_splits.main(args)

    verify_dataset_md5s(DATASETS.MIRAGE22)


def install_mirage19(input_folder=None, num_workers=30):
    from tcbench.libtcdatasets import mirage19_json_to_parquet
    from tcbench.libtcdatasets import mirage19_generate_splits

    rich_label("download & unpack")
    expected_files = [
        "MIRAGE-2019_traffic_dataset_downloadable_v2.tar.gz",
    ]
    # dataset_folder = _get_module_folder() / FOLDER_DATASETS / 'mirage19'
    dataset_folder = get_dataset_folder(DATASETS.MIRAGE19)

    raw_folder = dataset_folder / "raw"
    if input_folder:
        _verify_expected_files_exists(input_folder, expected_files)
        untar(input_folder / expected_files[0], raw_folder)
    else:
        datasets_yaml = load_datasets_yaml()
        url = datasets_yaml[str(DATASETS.MIRAGE19)]["data"]
        with tempfile.TemporaryDirectory() as tmpfolder:
            path = download_url(url, tmpfolder)
            untar(path, raw_folder)

    rich_label("preprocess", extra_new_line=True)
    preprocess_folder = dataset_folder / "preprocessed"
    cmd = f"--input-folder {raw_folder} --output-folder {preprocess_folder} --num-workers {num_workers}"
    args = mirage19_json_to_parquet.cli_parser().parse_args(cmd.split())
    mirage19_json_to_parquet.main(
        args.input_folder, args.output_folder / "mirage19.parquet", args.num_workers
    )

    rich_label("filter & generate splits", extra_new_line=True)
    # fooling the script believe there is a config.yml
    cmd = f"--config dummy_file.txt"
    args = mirage19_generate_splits.cli_parser().parse_args(cmd.split())
    args.config = dict(datasets={str(DATASETS.MIRAGE19): preprocess_folder})
    mirage19_generate_splits.main(args)

    verify_dataset_md5s(DATASETS.MIRAGE19)


def install(dataset_name, *args, **kwargs):
    dataset_name = str(dataset_name).replace("-", "_")
    curr_module = sys.modules[__name__]
    func_name = f"install_{dataset_name}"
    func = getattr(curr_module, func_name)
    return func(*args, **kwargs)


def get_dataset_parquet_filename(
    dataset_name: str | DATASETS, min_pkts: int = -1, split: str = None, animation=False
) -> pathlib.Path:
    """Returns the path of a dataset parquet file

    Arguments:
        dataset_name: The name of the dataset
        min_pkts: the filtering rule applied when curating the datasets.
            If -1, load the unfiltered dataset
        split: if min_pkts!=-1, is used to request the loading of
            the split file. For DATASETS.UCDAVISICDM19
            values can be "human", "script" or a number
            between 0 and 4.
            For all other dataset split can be anything
            which is not None (e.g., True)

    Returns:
        The pathlib.Path of a dataset parquet file
    """
    dataset_folder = get_dataset_folder(dataset_name) / "preprocessed"
    path = dataset_folder / f"{dataset_name}.parquet"

    if isinstance(split, int) and split < 0:
        split = None

    #    if isinstance(split, bool):
    #        split = 0
    #    elif isinstance(split, int):
    #        split = str(split)
    #
    #    if min_pkts == -1 and (split is None or int(split) < 0):
    #        return path
    #
    if isinstance(dataset_name, str):
        dataset_name = DATASETS.from_str(dataset_name)

    if dataset_name != DATASETS.UCDAVISICDM19:
        if min_pkts != -1:
            dataset_folder /= "imc23"
            if split is None:
                path = (
                    dataset_folder
                    / f"{dataset_name}_filtered_minpkts{min_pkts}.parquet"
                )
            else:
                path = (
                    dataset_folder
                    / f"{dataset_name}_filtered_minpkts{min_pkts}_splits.parquet"
                )
    else:
        #        if split is None:
        #            raise RuntimeError('split cannot be None for ucdavis-icdm19')
        #        dataset_folder /= 'imc23'
        #        if split in ('human', 'script'):
        #            path = dataset_folder / f'test_split_{split}.parquet'
        #        else:
        #            if split == 'train':
        #                split = 0
        #            path = dataset_folder / f'train_split_{split}.parquet'

        if split is not None:
            dataset_folder /= "imc23"
            if split in ("human", "script"):
                path = dataset_folder / f"test_split_{split}.parquet"
            else:
                if split == "train":
                    split = 0
                path = dataset_folder / f"train_split_{split}.parquet"

    return path


def load_parquet(
    dataset_name: str | DATASETS,
    min_pkts: int = -1,
    split: str = None,
    columns: List[str] = None,
    animation: bool = False,
) -> pd.DataFrame:
    """Load and returns a dataset parquet file

    Arguments:
        dataset_name: The name of the dataset
        min_pkts: the filtering rule applied when curating the datasets.
            If -1, load the unfiltered dataset
        split: if min_pkts!=-1, is used to request the loading of
            the split file. For DATASETS.UCDAVISICDM19
            values can be "human", "script" or a number
            between 0 and 4.
            For all other dataset split can be anything
            which is not None (e.g., True)
        columns: A list of columns to load (if None, load all columns)
        animation: if True, create a loading animation on the console

    Returns:
        A pandas dataframe and the related parquet file used to load the dataframe
    """
    path = get_dataset_parquet_filename(dataset_name, min_pkts, split)

    import pandas as pd
    from tcbench import cli

    if animation:
        with cli.console.status(f"loading: {path}...", spinner="dots"):
            return pd.read_parquet(path, columns=columns)
    return pd.read_parquet(path, columns=columns)


def get_split_indexes(dataset_name, min_pkts=-1):
    dataset_path = get_dataset_folder(dataset_name) / "preprocessed" / "imc23"
    if str(dataset_name) == str(DATASETS.UCDAVISICDM19):  #'ucdavis-icdm19':
        # automatically detect all split indexes
        split_indexes = sorted(
            [
                int(split_path.stem.rsplit("_", 1)[1])
                for split_path in dataset_path.glob("train_split*")
            ]
        )

    # elif args.dataset in (str(DATASETS.MIRAGE19), str(DATASETS.MIRAGE22), str(DATASETS.UTMOBILENET21)): #'mirage19', 'mirage22', 'utmobilenet21'):
    else:
        #        prefix = f'{args.dataset}_filtered'
        #        if args.dataset_minpkts != -1:
        #            prefix = f'{prefix}_minpkts{args.dataset_minpkts}'
        #        df_splits = pd.read_parquet(dataset_path / f'{prefix}_splits.parquet')
        #        split_indexes = df_splits['split_index'].unique().tolist()

        #    else:
        #        df_splits = pd.read_parquet(dataset_path / f'{args.dataset}_filtered_splits.parquet')
        #        df_splits = df_splits[df_splits['idx_inner_kfold'] == 0]
        #        split_indexes = list(map(int, df_splits['split_index'].values))
        df_splits = load_parquet(dataset_name, min_pkts=min_pkts, split=True)
        split_indexes = list(map(int, df_splits["split_index"].values))

    #    if args.max_train_splits == -1:
    #        args.max_train_splits = len(split_indexes)
    #
    #    split_indexes = split_indexes[:min(len(split_indexes), args.max_train_splits)]

    return split_indexes

def import_dataset(dataset_name, path_archive):
    data = load_datasets_yaml()
    folder_datasets = _get_module_folder() #/ FOLDER_DATASETS

    if dataset_name is None or str(dataset_name) not in data:
        raise RuntimeError(f"Invalid dataset name {dataset_name}")

    with tempfile.TemporaryDirectory() as tmpfolder:
        if path_archive is None:
            dataset_name = str(dataset_name)
            if "data_curated" not in data[dataset_name]:
                raise RuntimeError(f"The curated dataset cannot be downloaded (likely for licencing problems). Regenerate it using `tcbench datasets install --name {dataset_name}`")
            url = data[dataset_name]["data_curated"]
            expected_md5 = data[dataset_name]["data_curated_md5"]

            try:
                path_archive = download_url(url, tmpfolder)
            except requests.exceptions.SSLError:
                path_archive = download_url(url, tmpfolder, verify=False)

            md5 = get_md5(path_archive)
            assert md5 == expected_md5, f"MD5 check error: found {md5} while should be {expected_md5}"
        untar(path_archive, folder_datasets)

def verify_dataset_md5s(dataset_name):

    def flatten_dict(data):
        res = []
        for key, value in data.items():
            key = pathlib.Path(key)
            if isinstance(value, str):
                res.append((key, value))
                continue
            for inner_key, inner_value in flatten_dict(value):
                res.append((key / inner_key, inner_value))
        return res

    dataset_name = str(dataset_name)
    data_md5 = load_datasets_files_md5_yaml().get(dataset_name, None)
    expected_files = flatten_dict(data_md5)

    if dataset_name in (None, "") or data_md5 is None:
        raise RuntimeError(f"Invalid dataset name {dataset_name}")

    folder_dataset = _get_module_folder() / FOLDER_DATASETS / dataset_name
    if not folder_dataset.exists():
        raise RuntimeError(f"Dataset {dataset_name} is not installed. Run first \"tcbench datasets install --name {dataset_name}\"")

    folder_dataset /= "preprocessed"

    mismatches = dict()
    for exp_path, exp_md5 in richprogress.track(expected_files, description="Verifying parquet MD5..."):
        path = folder_dataset / exp_path
        if not path.exists():
            raise RuntimeError(f"File {path} not found")

        found_md5 = get_md5(path)
        fname = path.name
        if found_md5 == exp_md5:
            continue
        mismatches[path] = (exp_md5, found_md5)

    if mismatches:
        console.print(f"Found {len(mismatches)}/{len(expected_files)} mismatches when verifying parquet files md5")
        for path, (expected_md5, found_md5) in mismatches.items():
            console.print()
            console.print(f"path: {path}")
            console.print(f"expected_md5: {expected_md5}")
            console.print(f"found_md5: {found_md5}")
    else:
        console.print("All MD5 are correct!")
    
