import rich_click as click

import pathlib
import rich_click

from tcbench.cli import clickutils
from tcbench.cli.clickutils import (
    CLICK_TYPE_DATASET_NAME,
    CLICK_CALLBACK_DATASET_NAME,
    CLICK_TYPE_METHOD_NAME,
    CLICK_CALLBACK_METHOD_NAME,
    CLICK_CALLBACK_TOINT,
    CLICK_TYPE_INPUT_REPR,
    CLICK_CALLBACK_INPUT_REPR,
)

from tcbench import (
    DATASETS,
    DEFAULT_ARTIFACTS_FOLDER,
    DEFAULT_AIM_REPO,
    MODELING_METHOD_TYPE,
    MODELING_INPUT_REPR_TYPE,
    DEFAULT_CAMPAIGN_AUGATLOAD_AUGMENTATIONS,
    DEFAULT_CAMPAIGN_AUGATLOAD_SEEDS,
    DEFAULT_CAMPAIGN_AUGATLOAD_FLOWPICDIMS,
    DEFAULT_CAMPAIGN_CONTRALEARNANDFINETUNE_AUGMENTATIONS,
    DEFAULT_CAMPAING_CONTRALEARNANDFINETUNE_SEEDS_CONTRALEARN,
    DEFAULT_CAMPAIGN_CONTRALEARNANDFINETUNE_SEEDS_FINETUNE,
)

click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.USE_RICH_MARKUP = True

# VALID_AUGMENTATIONS_FOR_CONTRALEARN = "\[" + "|".join(
# name
# for name in DEFAULT_CAMPAIGN_AUGATLOAD_AUGMENTATIONS
# if name != "noaug"
# ) + "]."
#
# VALID_AUGMENTATIONS_FOR_AUGATLOAD =  '\[' +f'{"|".join(DEFAULT_CAMPAIGN_AUGATLOAD_AUGMENTATIONS)}' + "]."
VALID_AUGMENTATIONS_FOR_CONTRALEARN = clickutils.compose_help_string_from_list(
    DEFAULT_CAMPAIGN_AUGATLOAD_AUGMENTATIONS
)
VALID_AUGMENTATIONS_FOR_AUGATLOAD = clickutils.compose_help_string_from_list(
    DEFAULT_CAMPAIGN_AUGATLOAD_AUGMENTATIONS
)

OPTIONS_AUGMENT_AT_LOADING = [
    {
        "name": "General options",
        "options": [
            "--aim-experiment-name",
            "--aim-repo",
            "--artifacts-folder",
            "--campaign-id",
            "--dry-run",
            "--gpu-index",
            "--workers",
            "--seeds",
            "--help",
        ],
    },
    {
        "name": "Data",
        "options": [
            "--augmentations",
            "--dataset",
            "--dataset-minpkts",
            "--flowpic-dims",
            "--flowpic-block-duration",
            "--max-train-splits",
            "--split-indexes",
            "--train-val-split-ratio",
            "--no-test-leftover",
        ],
    },
    {
        "name": "Modeling",
        "options": ["--method"],
    },
    {
        "name": "DL hyper params",
        "options": [
            "--batch-size",
            "--epochs",
            "--learning-rate",
            "--patience-steps",
            "--no-dropout",
        ],
    },
    {
        "name": "XGBoost hyper params",
        "options": ["--input-repr", "--pktseries-len"],
    },
]

OPTIONS_CONTRALEARN_AND_FINETUNE = [
    {
        "name": "General options",
        "options": [
            "--aim-experiment-name",
            "--aim-repo",
            "--artifacts-folder",
            "--campaign-id",
            "--dry-run",
            "--gpu-index",
            "--workers",
            "--seeds",
            "--help",
        ],
    },
    {
        "name": "Data",
        "options": [
            "--augmentations",
            "--flowpic-dims",
            "--max-train-splits",
            "--split-indexes",
            "--train-val-split-ratio",
        ],
    },
    {
        "name": "Training hyperparams",
        "options": [
            "--batch-size",
            "--cl-projection-layer-dims",
            "--cl-seeds",
            "--ft-seeds",
            "--dropout",
        ],
    },
]


click.rich_click.OPTION_GROUPS.update(
    {
        "tcbench campaign augment-at-loading": OPTIONS_AUGMENT_AT_LOADING,
        "main.py campaign augment-at-loading": OPTIONS_AUGMENT_AT_LOADING,
        ##
        "tcbench campaign contralearn-and-finetune": OPTIONS_CONTRALEARN_AND_FINETUNE,
        "main.py campaign contralearn-and-finetune": OPTIONS_CONTRALEARN_AND_FINETUNE,
    }
)


@click.group("campaign")
@click.pass_context
def campaign(ctx):
    """Triggers a modeling campaign."""
    pass


@campaign.command("augment-at-loading")
@click.pass_context
@click.option(
    "--artifacts-folder",
    "artifacts_folder",
    type=pathlib.Path,
    default=DEFAULT_ARTIFACTS_FOLDER,
    show_default=True,
    help="Artifacts folder.",
)
@click.option(
    "--workers",
    "workers",
    type=int,
    default=20,
    show_default=True,
    help="Number of parallel worker for loading the data.",
)
@click.option(
    "--gpu-index",
    "gpu_index",
    type=str,
    default="0",
    show_default=True,
    help="The id of the GPU to use (if training with deep learning).",
)
@click.option(
    "--aim-repo",
    "aim_repo",
    type=pathlib.Path,
    default=DEFAULT_AIM_REPO,
    show_default=True,
    help="AIM repository location (local folder or URL).",
)
@click.option(
    "--aim-experiment-name",
    "aim_experiment_name",
    default="augmentations-at-loading",
    show_default=True,
    help="The name of the experiment for AIM tracking.",
)
###############
# flowpic
###############
@click.option(
    "--flowpic-dims",
    "flowpic_dims",
    type=str,
    default=",".join(map(str, DEFAULT_CAMPAIGN_AUGATLOAD_FLOWPICDIMS)),
    show_default=True,
    help="Coma separated list of flowpic dimensions for experiments.",
)
# @click.option(
#    "--flowpic-block-duration",
#    "flowpic_block_duration",
#    type=int,
#    default=15,
#    show_default=True,
#    help="Number of seconds for the head of a flow (i.e., block) to use for a flowpic.",
# )
###############
# data
###############
@click.option(
    "--dataset",
    "dataset",
    type=CLICK_TYPE_DATASET_NAME,
    callback=CLICK_CALLBACK_DATASET_NAME,
    default=str(DATASETS.UCDAVISICDM19),
    show_default=True,
    help="Dataset to use for modeling.",
)
@click.option(
    "--dataset-minpkts",
    type=click.Choice(("-1", "10", "100", "1000")),
    default="-1",
    callback=CLICK_CALLBACK_TOINT,
    show_default=True,
    help="In combination with --dataset, refines preprocessed and split dataset to use.",
)
@click.option(
    "--split-indexes",
    "split_indexes",
    type=str,
    default=None,
    help="Coma separted list of split indexes (by default all splits are used).",
)
@click.option(
    "--max-samples-per-class",
    type=int,
    default=-1,
    help="Activated when --split-indexes is -1 to define how many samples to select for train+val (with a 80/20 split between train and val).",
)
###############
# training
###############
# @click.option(
#    "--train-val-split-ratio",
#    "train_val_split_ratio",
#    type=float,
#    default=0.8,
#    show_default=True,
#    help="If not predefined by the selected split, the ratio data to use for training (rest is for validation).",
# )
@click.option(
    "--augmentations",
    type=str,
    default=",".join(map(str, DEFAULT_CAMPAIGN_AUGATLOAD_AUGMENTATIONS)),
    show_default=True,
    help="Coma separated list of augmentations for experiments. Choices: "
    + VALID_AUGMENTATIONS_FOR_AUGATLOAD,
)
@click.option(
    "--seeds",
    "seeds",
    type=str,
    default=",".join(map(str, DEFAULT_CAMPAIGN_AUGATLOAD_SEEDS)),
    show_default=True,
    help="Coma separated list of seed for experiments.",
)
@click.option(
    "--batch-size",
    "batch_size",
    type=int,
    default=32,
    show_default=True,
    help="Training batch size.",
)
@click.option(
    "--patience-steps",
    "patience_steps",
    default=5,
    type=int,
    show_default=True,
    help="Max. number of epochs without improvement before stopping training.",
)
@click.option(
    "--learning-rate",
    "learning_rate",
    type=float,
    default=0.001,
    show_default=True,
    help="Training learning rate.",
)
@click.option(
    "--epochs",
    "epochs",
    type=int,
    default=50,
    show_default=True,
    help="Number of epochs for training.",
)
@click.option(
    "--no-test-leftover",
    "suppress_test_train_val_leftover",
    default=False,
    is_flag=True,
    help="Skip test on leftover split (specific for ucdavis-icdm19, and default enabled for all other datasets).",
)
@click.option(
    "--no-dropout",
    "suppress_dropout",
    default=False,
    is_flag=True,
    help="Mask dropout layers with Identity layers.",
)
@click.option(
    "--method",
    "method",
    type=click.Choice(
        (str(MODELING_METHOD_TYPE.MONOLITHIC), str(MODELING_METHOD_TYPE.XGBOOST))
    ),
    default=str(MODELING_METHOD_TYPE.MONOLITHIC),
    show_default=True,
    help="Method to use for training.",
)
@click.option(
    "--input-repr",
    "flow_representation",
    type=CLICK_TYPE_INPUT_REPR,
    callback=CLICK_CALLBACK_INPUT_REPR,
    default=str(MODELING_INPUT_REPR_TYPE.PKTSERIES),
    show_default=True,
    metavar="TEXT",
    help="Input representation.",
)
@click.option(
    "--pktseries-len",
    "max_n_pkts",
    default="10,30",
    show_default=True,
    metavar="INTEGER",
    help="Number of packets (when using time series as input).",
)
@click.option(
    "--campaign-id",
    "campaign_id",
    type=str,
    default=None,
    help="A campaign id to mark all experiments.",
)
@click.option(
    "--dry-run",
    "dry_run",
    default=False,
    is_flag=True,
    help="Show the number of experiments and then quit.",
)
@click.option(
    "--max-train-splits",
    "max_train_splits",
    type=int,
    default=-1,
    show_default=True,
    help="The maximum number of training splits to experiment with. If -1, use all available.",
)
def augment_at_loading(ctx, **kwargs):
    """Modeling by applying data augmentation when loading the training set."""
    method = kwargs["method"]

    if method == str(MODELING_METHOD_TYPE.MONOLITHIC):
        from tcbench.modeling import (
            run_campaign_augmentations_at_loading as entry_point,
        )

        if str(kwargs["dataset"]) != str(DATASETS.UCDAVISICDM19):
            kwargs["suppress_test_train_val_leftover"] = True

        params = clickutils.convert_params_dict_to_list(
            kwargs, skip_params=["method", "flow_representation", "max_n_pkts"]
        )
    else:
        from tcbench.modeling import (
            run_campaign_augmentations_at_loading_xgboost as entry_point,
        )

        params = clickutils.convert_params_dict_to_list(
            kwargs,
            skip_params=[
                "method",
                "gpu_index",
                "augmentations",
                "batch_size",
                "learning_rate",
                "patience_steps",
                "epochs",
                "suppress_dropout",
                "dataset_minpkts",
                # "flowpic_dims",
                "dataset",
                "workers",
            ],
        )

    parser = entry_point.cli_parser()
    args = parser.parse_args((" ".join(params)).split())
    args.method = method
    entry_point.main(args)


@campaign.command("contralearn-and-finetune")
@click.pass_context
@click.option(
    "--campaign-id",
    "campaign_id",
    type=str,
    default=None,
    help="A campaign id to mark all experiments.",
)
@click.option(
    "--workers",
    "workers",
    type=int,
    default=50,
    show_default=True,
    help="Number of parallel worker for loading the data.",
)
@click.option(
    "--gpu-index",
    "gpu_index",
    type=str,
    default="0",
    show_default=True,
    help="The id of the GPU to use (if training with deep learning).",
)
@click.option(
    "--aim-repo",
    "aim_repo",
    type=pathlib.Path,
    default=DEFAULT_AIM_REPO,
    show_default=True,
    help="AIM repository location (local folder or URL).",
)
@click.option(
    "--aim-experiment-name",
    "aim_experiment_name",
    default="contrastive-learning-and-finetune",
    show_default=True,
    help="The name of the experiment for AIM tracking.",
)
@click.option(
    "--artifacts-folder",
    "artifacts_folder",
    type=pathlib.Path,
    default=DEFAULT_ARTIFACTS_FOLDER,
    show_default=True,
    help="Artifacts folder.",
)
@click.option(
    "--dry-run",
    "dry_run",
    default=False,
    is_flag=True,
    help="Show the number of experiments and then quit.",
)
@click.option(
    "--max-train-splits",
    "max_train_splits",
    type=int,
    default=-1,
    show_default=True,
    help="The maximum number of training splits to experiment with. If -1, use all available.",
)
@click.option(
    "--augmentations",
    type=str,
    default=DEFAULT_CAMPAIGN_CONTRALEARNANDFINETUNE_AUGMENTATIONS,
    show_default=True,
    help="Coma separated list of augmentations. Choices: "
    + VALID_AUGMENTATIONS_FOR_CONTRALEARN,
)
@click.option(
   "--train-val-split-ratio",
   "train_val_split_ratio",
   type=float,
   default=0.8,
   show_default=True,
   help="If not predefined by the selected split, the ratio data to use for training (rest is for validation).",
)
@click.option(
    "--flowpic-dims",
    "flowpic_dim",
    type=str,
    default="32",
    show_default=True,
    help="Coma separated list of flowpic dimensions for experiments.",
)
@click.option(
    "--cl-seeds",
    "contrastive_learning_seeds",
    default=",".join(
        map(
            str,
            DEFAULT_CAMPAING_CONTRALEARNANDFINETUNE_SEEDS_CONTRALEARN,
        )
    ),
    show_default=True,
    help="Coma separated list of seeds to use for contrastive learning pretraining.",
)
@click.option(
    "--ft-seeds",
    "finetune_seeds",
    default=",".join(map(str, DEFAULT_CAMPAIGN_CONTRALEARNANDFINETUNE_SEEDS_FINETUNE)),
    show_default=True,
    help="Coma separated list of seeds to use for finetune training.",
)
@click.option(
    "--dropout",
    "dropout",
    default="disabled",
    show_default=True,
    help="Coma separated list. Choices:"
    + clickutils.compose_help_string_from_list(("enabled", "disabled")),
)
@click.option(
    "--cl-projection-layer-dims",
    "projection_layer_dims",
    type=str,
    default="30",
    help="Coma separate list of contrastive learning projection layer dimensions.",
    show_default=True,
)
#    parser.add_argument(
#        "--finetune-augmentation", default="none",
#        choices=("none", "only-views", "views-and-original"),
#        help=utils.compose_cli_help_string("Optional augmentation for finetuning training data. With 'only-views' finetuning is performed only using augmented data; with 'views-and-original' finetuning is performed using augmentation and original data. By default, no augmentation is performed")
#    )
@click.option(
    "--batch-size",
    "batch_size",
    type=int,
    default=32,
    show_default=True,
    help="Training batch size.",
)
@click.option(
    "--split-indexes",
    "split_indexes",
    type=str,
    default=None,
    help="Coma separted list of split indexes (by default all splits are used).",
)
def contrastivelearning_and_finetune(ctx, **kwargs):
    """Modeling by pre-training via constrative learning and then finetune the final classifier from the pre-trained model."""
    from tcbench.modeling import (
        run_campaign_contrastive_learning_and_finetune as entry_point,
    )

    kwargs["dataset"] = DATASETS.UCDAVISICDM19

    if str(kwargs["dataset"]) != str(DATASETS.UCDAVISICDM19):
        kwargs["suppress_test_train_val_leftover"] = True

    params = clickutils.convert_params_dict_to_list(
        kwargs,
        skip_params=["dataset"],
    )
    parser = entry_point.cli_parser()
    args = parser.parse_args((" ".join(params)).split())
    entry_point.main(args)
