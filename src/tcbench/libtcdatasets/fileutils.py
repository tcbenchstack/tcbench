from __future__ import annotations
from typing import Dict, Any

import rich.progress as richprogress

import yaml
import pathlib
import hashlib
import sys
import tarfile
import zipfile
import requests
import os
import shutil

# import tcbench.cli
from tcbench.libtcdatasets.constants import *
from tcbench.cli import richutils


def load_yaml(fname: pathlib.Path) -> Dict[Any, Any]:
    """Load an input YAML file

    Arguments:
        fname: the YAML filename to load

    Return:
        The YAML object loaded
    """
    with open(fname) as fin:
        return yaml.safe_load(fin)


def save_yaml(data: Any, save_as: pathlib.Path) -> None:
    save_as = pathlib.Path(save_as)
    if not save_as.parent.exists():
        save_as.parent.mkdir(parents=True)
    with open(save_as, "w") as fout:
        yaml.dump(data, fout)


def load_config(fname: pathlib.Path) -> Dict:
    """Load the configuration file of the framework

    Arguments:
        fname: the YAML config file to load

    Return:
        The loaded config file
    """
    return load_yaml(fname)


def get_md5(path: pathlib.Path) -> str:
    h = hashlib.new("md5")
    with open(str(path), "rb") as fin:
        h.update(fin.read())
    return h.hexdigest()


def get_module_folder():
    curr_module = sys.modules[__name__]
    folder = pathlib.Path(curr_module.__file__).parent
    return folder


# def get_rich_tree_datasets_properties(dataset_name=None):
#    data = load_datasets_yaml()
#    folder_datasets = get_module_folder() / FOLDER_DATASETS
#
#    if dataset_name:
#        dataset_name = str(dataset_name)
#
#    tree = Tree("Datasets")
#    for curr_dataset_name, attributes in data.items():
#        if dataset_name is not None and curr_dataset_name != dataset_name:
#            continue
#
#        node = tree.add(curr_dataset_name)
#
#        curr_dataset_folder = folder_datasets / curr_dataset_name
#        is_installed = (curr_dataset_folder / "raw").exists()
#        is_preprocessed = (
#            curr_dataset_folder / "preprocessed" / f"{curr_dataset_name}.parquet"
#        ).exists()
#        has_splits = (curr_dataset_folder / "preprocessed" / "imc23").exists()
#
#        table = Table(show_header=False, box=None)
#        table.add_column("property")
#        table.add_column("value", overflow="fold")
#
#        table.add_row(":triangular_flag: classes:", str(attributes["num_classes"]))
#        table.add_row(":link: paper_url:", attributes["paper"])
#        table.add_row(":link: website:", attributes["website"])
#        table.add_row(":link: data:", attributes["data"])
#        if "data_curated" in attributes:
#            table.add_row(":link: curated data:", attributes["data_curated"])
#            table.add_row(":heavy_plus_sign: curated data MD5:", attributes["data_curated_md5"])
#
#        if is_installed:
#            path = curr_dataset_folder / "raw"
#            text = f"[green]{path}[/green]"
#        else:
#            text = "[red]None[/red]"
#        table.add_row(":file_folder: installed:", text)
#
#        if is_preprocessed:
#            path = curr_dataset_folder / "preprocessed"
#            text = f"[green]{path}[/green]"
#        else:
#            text = "[red]None[/red]"
#        table.add_row(":file_folder: preprocessed:", text)
#
#        if has_splits:
#            path = curr_dataset_folder / "preprocessed" / "imc23"
#            text = f"[green]{path}[/green]"
#        else:
#            text = f"[red]None[/red]"
#        table.add_row(":file_folder: data splits:", text)
#
#        node.add(table)
#
#    return tree


# def get_rich_tree_parquet_files(dataset_name=None):
#    data = load_datasets_yaml()
#    folder_datasets = get_module_folder() / FOLDER_DATASETS
#
#    if dataset_name:
#        dataset_name = str(dataset_name)
#
#    tree = Tree("Datasets")
#    for curr_dataset_name, attributes in data.items():
#        if dataset_name is not None and curr_dataset_name != dataset_name:
#            continue
#
#        node = tree.add(curr_dataset_name)
#
#        preprocessed = Tree(":file_folder: preprocessed/")
#        preprocessed.add(f"{curr_dataset_name}.parquet")
#
#        path = folder_datasets / curr_dataset_name / "preprocessed" / "LICENSE"
#        if path.exists():
#            preprocessed.add("LICENSE")
#
#        imc23 = Tree(":file_folder: imc23/")
#        folder = folder_datasets / curr_dataset_name / "preprocessed" / "imc23"
#        for path in sorted(folder.glob("*.parquet")):
#            imc23.add(path.name)
#        preprocessed.add(imc23)
#
#        node.add(preprocessed)
#    return tree


# def get_rich_dataset_schema(dataset_name, schema_type):
#    folder = get_dataset_resources_folder()
#    path = folder / f"{dataset_name}.yml"
#    data = load_yaml(path)
#    if dataset_name == DATASETS.UCDAVISICDM19:
#        schema_type = "__all__"
#    else:
#        schema_type = f"__{schema_type}__"
#    schema = data[schema_type]
#    table = Table()
#    table.add_column("Field")
#    table.add_column("Dtype")
#    table.add_column("Description", overflow="fold")
#    for name, attrs in schema.items():
#        table.add_row(name, attrs["dtype"], attrs["description"])
#    return table


#def download_url(
#    url: str,
#    save_to: pathlib.Path,
#    verify_tls: bool = True,
#    force_redownload: bool = False,
#) -> pathlib.Path:
#    """Download content via URL.
#
#    Args:
#        url: the object URL.
#        save_to: the destination folder.
#        verify_tls: if False, skip TLS verification when downloading.
#        force_redownload: if False, return with no action if the destination folder
#            already contains a file with the expected name
#
#    Returns:
#        the path of the downloaded file
#    """
#    # from tcbench.cli import get_rich_console
#    save_to = pathlib.Path(save_to)
#
#    fname = pathlib.Path(url).name
#    save_as = save_to / fname
#    if save_as.exists() and not force_redownload:
#        return save_as
#
#    resp = requests.get(url, stream=True, verify=verify_tls)
#    totalbytes = int(resp.headers.get("content-length", 0))
#
#    if not save_as.parent.exists():
#        save_as.parent.mkdir(parents=True)
#
#    with (
#        richutils.FileDownloadProgress(totalbytes=totalbytes) as progressbar,
#        open(str(save_as), "wb") as fout,
#    ):
#        for data in resp.iter_content(chunk_size=64 * 1024):
#            size = fout.write(data)
#            progressbar.update(advance=size)
#
#    return save_as
#
#
#def _verify_expected_files_exists(folder, expected_files):
#    for fname in expected_files:
#        path = folder / fname
#        if not path.exists():
#            raise RuntimeError(f"missing {path}")
#
#
#def unzip(src: str | pathlib.Path, dst: str | pathlib.Path = None) -> pathlib.Path:
#    """Unpack a zip archive.
#
#    Arguments:
#        src: path of the .zip archive
#        dst: destination folder. If None, the archive is
#            unpacked in the same folder of src
#
#    Returns:
#        the destination folder
#    """
#    src = pathlib.Path(src)
#    if dst is None:
#        dst = src.parent
#    else:
#        dst = pathlib.Path(dst)
#
#    if dst.exists():
#        shutil.rmtree(dst, ignore_errors=True)
#    dst.mkdir(parents=True)
#
#    with richutils.SpinnerProgress(f"Unpacking..."), zipfile.ZipFile(src) as fzipped:
#        fzipped.extractall(dst)
#    return dst
#
#
#def untar(src: pathlib.Path, dst: pathlib.Path = None) -> pathlib.Path:
#    """Unpack a tarball archive.
#
#    Arguments:
#        src: path of the .zip archive
#        dst: destination folder. If None, the archive is
#            unpacked in the same folder of src
#
#    Returns:
#        the destination folder
#    """
#    src = pathlib.Path(src)
#    if dst is None:
#        dst = src.parent
#    else:
#        dst = pathlib.Path(dst)
#
#    if dst.exists():
#        shutil.rmtree(dst, ignore_errors=True)
#    dst.mkdir(parents=True)
#
#    with richutils.SpinnerProgress(f"Unpacking..."), tarfile.open(src, "r:gz") as ftar:
#        ftar.extractall(dst)
#    return dst
