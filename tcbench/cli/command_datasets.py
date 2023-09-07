import rich_click as click

import requests
import yaml
import pathlib
import sys
import shutil

from tcbench.libtcdatasets import datasets_utils
from tcbench import cli
from tcbench.cli.clickutils import (
    CLICK_TYPE_DATASET_NAME,
    CLICK_CALLBACK_DATASET_NAME,
    CLICK_CALLBACK_TOINT,
)
from tcbench.cli import richutils


@click.group()
@click.pass_context
def datasets(ctx):
    """Install/Remove traffic classification datasets."""
    pass


@datasets.command(name="info")
@click.pass_context
@click.option(
    "--name",
    "-n",
    "dataset_name",
    required=False,
    type=CLICK_TYPE_DATASET_NAME,
    callback=CLICK_CALLBACK_DATASET_NAME,
    help="Dataset to install.",
    default=None,
)
def info(ctx, dataset_name):
    """Show the meta-data related to supported datasets."""
    rich_obj = datasets_utils.get_rich_tree_datasets_properties(dataset_name)
    cli.console.print(rich_obj)


@datasets.command(name="install")
@click.pass_context
@click.option(
    "--name",
    "-n",
    "dataset_name",
    required=True,
    type=CLICK_TYPE_DATASET_NAME,
    callback=CLICK_CALLBACK_DATASET_NAME,
    help="Dataset to install.",
)
@click.option(
    "--input-folder",
    "-i",
    "input_folder",
    required=False,
    type=pathlib.Path,
    default=None,
    help="Folder where to find pre-downloaded tarballs.",
)
def install(ctx, dataset_name, input_folder):
    """Install a dataset."""
    if (
        dataset_name
        in (
            datasets_utils.DATASETS.UCDAVISICDM19,
            datasets_utils.DATASETS.UTMOBILENET21,
        )
        and input_folder is None
    ):
        raise RuntimeError(
            f"Cannot automatically download {dataset_name}. Please download it separately and retry install using the --input-folder option"
        )
    datasets_utils.install(dataset_name, input_folder)


@datasets.command(name="lsparquet")
@click.pass_context
@click.option(
    "--name",
    "-n",
    "dataset_name",
    required=False,
    type=CLICK_TYPE_DATASET_NAME,
    callback=CLICK_CALLBACK_DATASET_NAME,
    default=None,
    help="Dataset to install.",
)
def lsparquet(ctx, dataset_name):
    """Tree view of the datasets parquet files."""
    rich_obj = datasets_utils.get_rich_tree_parquet_files(dataset_name)
    cli.console.print(rich_obj)


@datasets.command(name="schema")
@click.pass_context
@click.option(
    "--name",
    "-n",
    "dataset_name",
    required=False,
    type=CLICK_TYPE_DATASET_NAME,
    callback=CLICK_CALLBACK_DATASET_NAME,
    default=None,
    help="Dataset to install.",
)
@click.option(
    "--type",
    "-t",
    "schema_type",
    required=False,
    type=click.Choice(("unfiltered", "filtered", "splits")),
    default="unfiltered",
    help="Schema type (unfiltered: original raw data; filtered: curated data; splits: train/val/test splits).",
)
def schema(ctx, dataset_name, schema_type):
    """Show datasets schemas"""
    rich_obj = datasets_utils.get_rich_dataset_schema(dataset_name, schema_type)
    cli.console.print(rich_obj)


@datasets.command(name="samples-count")
@click.pass_context
@click.option(
    "--name",
    "-n",
    "dataset_name",
    required=False,
    type=CLICK_TYPE_DATASET_NAME,
    callback=CLICK_CALLBACK_DATASET_NAME,
    default=None,
    help="Dataset to install.",
)
@click.option(
    "--min-pkts",
    "min_pkts",
    required=False,
    type=click.Choice(("-1", "10", "1000")),
    default="-1",
    help="",
)
@click.option(
    "--split",
    "split",
    required=False,
    type=click.Choice(("human", "script", "0", "1", "2", "3", "4")),
    default=None,
    help="",
)
def report_samples_count(ctx, dataset_name, min_pkts, split):
    """Show report on number of samples per class."""
    with cli.console.status("Computing...", spinner="dots"):
        min_pkts = int(min_pkts)
        if min_pkts == -1 and split is not None:
            if dataset_name != datasets_utils.DATASETS.UCDAVISICDM19:
                min_pkts = 10

        df_split = None
        if dataset_name == datasets_utils.DATASETS.UCDAVISICDM19 or split is None:
            df = datasets_utils.load_parquet(dataset_name, min_pkts, split)
        else:
            df = datasets_utils.load_parquet(dataset_name, min_pkts, split=None)
            df_split = datasets_utils.load_parquet(dataset_name, min_pkts, split=split)

    title = "unfiltered"
    if dataset_name == datasets_utils.DATASETS.UCDAVISICDM19:
        if split is not None:
            title = f"filtered, split: {split}"
    else:
        title = []
        if min_pkts != -1:
            title.append(f"min_pkts: {min_pkts}")
        if split:
            title.append(f"split: {split}")
        if title:
            title = ", ".join(title)
        else:
            title = "unfiltered"

    if df_split is None:
        if (
            dataset_name == datasets_utils.DATASETS.UCDAVISICDM19
            and min_pkts == -1
            and split is None
        ):
            ser = df.groupby(["partition", "app"])["app"].count()
        else:
            ser = df["app"].value_counts()

        richutils.rich_samples_count_report(ser, title=title)
    else:
        richutils.rich_splits_report(df, df_split, split_index=split, title=title)


@datasets.command(name="import")
@click.pass_context
@click.option(
    "--input-folder",
    "-i",
    "input_folder",
    required=False,
    type=pathlib.Path,
    default=None,
    help="Folder where to find pre-computed curated datasets.",
)
def import_datasets(ctx, input_folder):
    """Import datasets."""
    subfolders = list(input_folder.iterdir())
    if len(subfolders) == 1 and subfolders[0].name == "datasets":
        input_folder /= "datasets"

    datasets_root_folder = datasets_utils.get_datasets_root_folder()
    for item in input_folder.iterdir():
        if not item.is_dir():
            continue

        src = item
        dst = datasets_root_folder / item.name
        with cli.console.status(f"Importing {item.name}...", spinner="dots"):
            shutil.copytree(str(src), str(dst))


@datasets.command(name="delete")
@click.pass_context
@click.option(
    "--name",
    "-n",
    "dataset_name",
    required=False,
    type=CLICK_TYPE_DATASET_NAME,
    callback=CLICK_CALLBACK_DATASET_NAME,
    default=None,
    help="Dataset to delete.",
)
def delete_dataset(ctx, dataset_name):
    """Delete a dataset."""
    folder = datasets_utils.get_dataset_folder(dataset_name)
    if not folder.exists():
        cli.console.print(f"[red]Dataset {dataset_name} is not installed[/red]")
    else:
        with cli.console.status(f"Deleting {dataset_name}...", spinner="dots"):
            shutil.rmtree(str(folder))
