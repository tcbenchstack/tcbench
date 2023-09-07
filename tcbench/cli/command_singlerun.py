import rich_click as click

import pathlib

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
)

click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.USE_RICH_MARKUP = True


OPTIONS_AUGMENT_AT_LOADING = [
    {
        "name": "General options",
        "options": [
            "--aim-experiment-name",
            "--aim-repo",
            "--artifacts-folder",
            "--gpu-index",
            "--workers",
            "--seed",
            "--help",
        ],
    },
    {
        "name": "Data",
        "options": [
            "--dataset",
            "--dataset-minpkts",
            "--flowpic-dim",
            "--flowpic-block-duration",
            "--split-index",
            "--train-val-split-ratio",
            "--aug-name",
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
            "--learning-rate",
            "--patience-steps",
            "--epochs",
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
            "--gpu-index",
            "--workers",
            "--help",
        ],
    },
    {
        "name": "Data",
        "options": [
            "--dataset",
            "--flowpic-dim",
            "--flowpic-block-duration",
            "--split-index",
            "--no-test-leftover",
        ],
    },
    {
        "name": "General Deeplearning hyperparams",
        "options": [
            "--batch-size",
            "--no-dropout",
        ],
    },
    {
        "name": "Contrastive learning hyperparams",
        "options": [
            "--cl-aug-names",
            "--cl-projection-layer-dim",
            "--cl-learning-rate",
            "--cl-seed",
            "--cl-patience-steps",
            "--cl-temperature",
            "--cl-epochs",
        ],
    },
    {
        "name": "Finetune hyperparams",
        "options": [
            "--ft-learning-rate",
            "--ft-patience-steps",
            "--ft-patience-min-delta",
            "--ft-train-samples",
            "--ft-epochs",
            "--ft-seed",
        ],
    },
]

click.rich_click.OPTION_GROUPS.update(
    {
        "tcbench run augment-at-loading": OPTIONS_AUGMENT_AT_LOADING,
        "main.py run augment-at-loading": OPTIONS_AUGMENT_AT_LOADING,
        ##
        "tcbench run contralearn-and-finetune": OPTIONS_CONTRALEARN_AND_FINETUNE,
        "main.py run contralearn-and-finetune": OPTIONS_CONTRALEARN_AND_FINETUNE,
    }
)


@click.group("run")
@click.pass_context
def singlerun(ctx):
    """Triggers a modeling run."""
    pass


#######################################
# AUGMENT_AT_LOADING
#######################################


@singlerun.command("augment-at-loading")
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
    default="augmentation-at-loading",
    show_default=True,
    help="The name of the experiment for AIM tracking.",
)
###############
# flowpic
###############
@click.option(
    "--flowpic-dim",
    "flowpic_dim",
    type=click.Choice(("32", "64", "1500")),
    callback=CLICK_CALLBACK_TOINT,
    default="32",
    show_default=True,
    help="Flowpic dimension.",
)
@click.option(
    "--flowpic-block-duration",
    "flowpic_block_duration",
    type=int,
    default=15,
    show_default=True,
    help="Number of seconds for the head of a flow (i.e., block) to use for a flowpic.",
)
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
    "--split-index",
    "split_index",
    type=int,
    default=0,
    show_default=True,
    help="Data split index.",
)
#    parser.add_argument(
#        "--max-samples-per-class",
#        type=int,
#        default=-1,
#        help=utils.compose_cli_help_string("Activated when --split-index is -1 to define how many samples to select for train+val (with a 80/20 split between train and val")
#    )
###############
# training
###############
@click.option(
    "--train-val-split-ratio",
    "train_val_split_ratio",
    type=float,
    default=0.8,
    show_default=True,
    help="If not predefined by the selected split, the ratio data to use for training (rest is for validation).",
)
@click.option(
    "--aug-name",
    "aug_name",
    type=click.Choice(
        (
            "noaug",
            "rotate",
            "horizontalflip",
            "colorjitter",
            "packetloss",
            "timeshift",
            "changertt",
        )
    ),
    default="noaug",
    show_default=True,
    help="Name of the augmentation to use.",
)
#    parser.add_argument(
#        "--suppress-val-augmentation",
#        action='store_true',
#        default=False,
#        help=utils.compose_cli_help_string('Do not augment validation set')
#    )
@click.option(
    "--seed",
    "seed",
    type=int,
    default=12345,
    show_default=True,
    help="Seed to initialize random generators.",
)
@click.option(
    "--batch-size",
    "batch_size",
    type=int,
    default=32,
    show_default=True,
    help="Training batch size",
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
    #metavar="TEXT",
    help="Input representation.",
)
@click.option(
    "--pktseries-len",
    "max_n_pkts",
    type=click.Choice(("10", "30")),
    default="10",
    show_default=True,
    metavar="INTEGER",
    help="Number of packets (when using time series as input).",
)
def augment_at_loading(ctx, **kwargs):
    """Modeling by applying data augmentation when loading the training set."""
    method = kwargs["method"]

    if method == str(MODELING_METHOD_TYPE.MONOLITHIC):
        from tcbench.modeling import run_augmentations_at_loading as entry_point

        if str(kwargs["dataset"]) != str(DATASETS.UCDAVISICDM19):
            kwargs["suppress_test_train_val_leftover"] = True

        params = clickutils.convert_params_dict_to_list(
            kwargs, skip_params=["method", "flow_representation", "max_n_pkts"]
        )
    else:
        from tcbench.modeling import run_augmentations_at_loading_xgboost as entry_point

        params = clickutils.convert_params_dict_to_list(
            kwargs,
            skip_params=[
                "method",
                "gpu_index",
                "aug_name",
                "batch_size",
                "learning_rate",
                "patience_steps",
                "epochs",
                "suppress_dropout",
                "dataset_minpkts",
            ],
        )

    parser = entry_point.cli_parser()
    args = parser.parse_args((" ".join(params)).split())
    args.method = method
    entry_point.main(args)


#######################################
# CONTRASTIVE LEARNING AND FINETUNE
#######################################


@singlerun.command("contralearn-and-finetune")
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
    default="contrastive-learning-and-finetune",
    show_default=True,
    help="The name of the experiment for AIM tracking.",
)
#    parser.add_argument("--final", action="store_true", default=False)
###############
# flowpic
###############
@click.option(
    "--flowpic-dim",
    "flowpic_dim",
    type=click.Choice(("32",)),  # "64", "1500")),
    callback=CLICK_CALLBACK_TOINT,
    default="32",
    show_default=True,
    help="Flowpic dimension.",
)
@click.option(
    "--flowpic-block-duration",
    "flowpic_block_duration",
    type=int,
    default=15,
    show_default=True,
    help="Number of seconds for the head of a flow (i.e., block) to use for a flowpic.",
)
###############
# data
###############
@click.option(
    "--dataset",
    "dataset",
    type=click.Choice((str(DATASETS.UCDAVISICDM19),)),  # CLICK_TYPE_DATASET_NAME,
    # callback=CLICK_CALLBACK_DATASET_NAME,
    default=str(DATASETS.UCDAVISICDM19),
    show_default=True,
    help="Dataset to use for modeling.",
)
# @click.option(
#    "--dataset-minpkts",
#    type=click.Choice(("-1", "10", "100", "1000")),
#    default="-1",
#    callback=CLICK_CALLBACK_TOINT,
#    show_default=True,
#    help="In combination with --dataset, refines preprocessed and split dataset to use.",
# )
@click.option(
    "--split-index",
    "split_index",
    type=int,
    default=0,
    show_default=True,
    help="Data split index.",
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
# @click.option(
#    "--aug-name",
#    "aug_name",
#    type=click.Choice(
#        (
#            "noaug",
#            "rotate",
#            "horizontalflip",
#            "colorjitter",
#            "packetloss",
#            "timeshift",
#            "changertt",
#        )
#    ),
#    default="noaug",
#    show_default=True,
#    help="Name of the augmentation to use.",
# )
#    parser.add_argument(
#        "--suppress-val-augmentation",
#        action='store_true',
#        default=False,
#        help=utils.compose_cli_help_string('Do not augment validation set')
#    )
@click.option(
    "--batch-size",
    "batch_size",
    type=int,
    default=32,
    show_default=True,
    help="Training batch size",
)
@click.option(
    "--no-dropout",
    "suppress_dropout",
    default=False,
    is_flag=True,
    help="Mask dropout layers with Identity layers.",
)
# @click.option(
#    "--method",
#    "method",
#    type=click.Choice(
#        (str(MODELING_METHOD_TYPE.MONOLITHIC), str(MODELING_METHOD_TYPE.XGBOOST))
#    ),
#    default=str(MODELING_METHOD_TYPE.MONOLITHIC),
#    show_default=True,
#    help="Method to use for training.",
# )
@click.option(
    "--cl-aug-names",
    "augmentations",
    default="changertt,timeshift",
    show_default=True,
    help="Coma separated list of augmentations pool for contrastive learning.",
)
@click.option(
    "--cl-projection-layer-dim",
    "projection_layer_dim",
    type=int,
    default=30,
    help="The number of units in the contrastive learning projection layer.",
    show_default=True,
)
@click.option(
    "--cl-learning-rate",
    "contrastive_learning_lr",
    type=float,
    default=0.001,
    show_default=True,
    help="Learning rate for pretraining.",
)
@click.option(
    "--cl-seed",
    "contrastive_learning_seed",
    type=int,
    default=12345,
    show_default=True,
    help="Seed for contrastive learning pretraining.",
)
@click.option(
    "--cl-patience-steps",
    "contrastive_learning_patience_steps",
    type=int,
    default=3,
    help="Max steps to wait before stopping training if the top5 validation accuracy does not improve.",
    show_default=True,
)
@click.option(
    "--cl-temperature",
    "contrastive_learning_temperature",
    type=float,
    default=0.07,
    help="Temperature for InfoNCE loss.",
    show_default=True,
)
@click.option(
    "--cl-epochs",
    "contrastive_learning_epochs",
    type=int,
    default=50,
    show_default=True,
    help="Epochs for contrastive learning pretraining.",
)
####################################
# finetune configs
####################################
@click.option(
    "--ft-learning-rate",
    "finetune_lr",
    type=float,
    default=0.01,
    help="Learning rate for finetune.",
    show_default=True,
)
@click.option(
    "--ft-patience-steps",
    "finetune_patience_steps",
    type=int,
    default=5,
    show_default=True,
    help="Max steps to wait before stopping finetune training loss does not improve.",
)
@click.option(
    "--ft-patience-min-delta",
    "finetune_patience_min_delta",
    type=float,
    default=0.001,
    show_default=True,
    help="Minimum decrease of training loss to be considered as improvement.",
)
@click.option(
    "--ft-train-samples",
    "finetune_train_samples",
    type=int,
    default=10,
    show_default=True,
    help="Number of samples per-class for finetune training.",
)
@click.option(
    "--ft-epochs",
    "finetune_epochs",
    type=int,
    default=50,
    show_default=True,
    help="Epochs for finetune training.",
)
@click.option(
    "--ft-seed",
    "finetune_seed",
    type=int,
    default=12345,
    show_default=True,
    help="Seed for finetune training.",
)
def contrastivelearning_and_finetune(ctx, **kwargs):
    """Modeling by pre-training via constrative learning and then finetune the final classifier from the pre-trained model."""

    import tcbench.modeling.run_contrastive_learning_and_finetune as entry_point

    params = clickutils.convert_params_dict_to_list(
        kwargs,  # skip_params=["method", "flow_representation", "max_n_pkts"]
    )

    parser = entry_point.cli_parser()
    args = parser.parse_args((" ".join(params)).split())
    args.method = "simclr"
    args.augmentations = kwargs["augmentations"].split(",")
    entry_point.main(args)
