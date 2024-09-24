from __future__ import annotations
import rich_click as click

from typing import Iterable
import pathlib

import tcbench
from tcbench.cli.clickutils import (
    DATASET_NAME,
    DATASET_TYPE,
    MODELING_METHOD_NAME,
    CLICK_CHOICE_DATASET_NAME,
    CLICK_CHOICE_DATASET_TYPE,
    CLICK_CHOICE_MODELING_METHOD_NAME,
    CLICK_PARSE_DATASET_NAME,
    CLICK_PARSE_DATASET_TYPE,
    CLICK_PARSE_MODELING_METHOD_NAME,
    CLICK_PARSE_STR_TO_LIST_INT,
)
from tcbench.modeling.ml import loops
from tcbench.modeling.ml.core import MultiClassificationResults
from tcbench.modeling import (
    MODELING_FEATURE
)

@click.group()
@click.pass_context
def modeling(ctx):
    """Handle modeling experiments."""
    pass


@modeling.command(name="run")
@click.option(
    "--dset-name",
    "-d",
    "dataset_name",
    required=False,
    type=CLICK_CHOICE_DATASET_NAME,
    callback=CLICK_PARSE_DATASET_NAME,
    help="Dataset name.",
    default=None,
)
@click.option(
    "--dset-type",
    "-t",
    "dataset_type",
    required=False,
    type=CLICK_CHOICE_DATASET_TYPE,
    callback=CLICK_PARSE_DATASET_TYPE,
    help="Dataset type.",
    #default=click.Choice(DATASET_TYPE.CURATE),
)
@click.option(
    "--method",
    "-m",
    "method_name",
    required=False,
    type=CLICK_CHOICE_MODELING_METHOD_NAME,
    callback=CLICK_PARSE_MODELING_METHOD_NAME,
    help="Modeling method.",
    default=None,
)
@click.option(
    "--series-len",
    "-s",
    "series_len",
    required=True,
    type=int,
    help="Clip packet series to the specified length.",
    default=10,
)
@click.option(
    "--output-folder",
    "-o",
    "save_to",
    required=False,
    default=pathlib.Path("./model"),
    type=pathlib.Path,
    help="Output folder."
)
@click.option(
    "--workers",
    "-w",
    "num_workers",
    required=False,
    default=1,
    type=int,
    help="Number of parallel workers."
)
@click.option(
    "--split-indices",
    "-i",
    "split_indices",
    required=False,
    default="",
    type=tuple,
    callback=CLICK_PARSE_STR_TO_LIST_INT,
    help="Number of parallel workers."
)
@click.pass_context
def run(
    ctx, 
    dataset_name: DATASET_NAME, 
    dataset_type: DATASET_TYPE,
    method_name: MODELING_METHOD_NAME,
    series_len: int,
    save_to: pathlib.Path,
    num_workers: int,
    split_indices: Iterable[int],
) -> Iterable[MultiClassificationResults]:
    """Run an experiment or campaign."""
    return loops.train_loop(
        dataset_name=dataset_name,
        dataset_type=dataset_type,
        method_name=method_name,
        series_len=series_len,
        features=(
            MODELING_FEATURE.PKTS_SIZE, 
            MODELING_FEATURE.PKTS_DIR,
        ),
        save_to=save_to,
        num_workers=num_workers,
        split_indices=split_indices,
    )

