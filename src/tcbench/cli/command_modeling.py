from __future__ import annotations
import rich_click as click

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
)
from tcbench.modeling.ml.loops as ml_loops
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
    default=None,
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
@click.pass_context
def run(
    ctx, 
    dataset_name: DATASET_NAME, 
    dataset_type: DATASET_TYPE,
    method_name: MODELING_METHOD_NAME,
):
    """Run an experiment or campaign."""
    catalog = tcbench.datasets_catalog()
    dset = catalog[dataset_name].load(dataset_type)

    df_splits = splitting.split_monte_carlo(
        dset.df,
        y_colname = dset.y_colname,
        index_colname = dset.index_colname, 
        num_splits = 1,
        seed = 1,
        test_size = 0.1,
    )

    ml_loops.train_loop(
        df,
        df_splits,
        features = [MODELING_FEATURE.PKTS_SIZE, MODELIGN_FEATURE.PKTS_DIR],
        series_len = 10,
    )

