from datetime import datetime, timedelta

import itertools
import argparse
import pathlib
import time
import sys

from tcbench import (
    DATASETS,
    DEFAULT_CAMPAIGN_CONTRALEARNANDFINETUNE_FLOWPICDIMS,
    DEFAULT_CAMPAING_CONTRALEARNANDFINETUNE_SEEDS_CONTRALEARN,
    DEFAULT_CAMPAIGN_CONTRALEARNANDFINETUNE_SEEDS_FINETUNE,
    DEFAULT_CAMPAIGN_CONTRALEARNANDFINETUNE_AUGMENTATIONS,
    DEFAULT_CAMPAIGN_CONTRALEARNANDFINETUNE_VALID_AUGMENTATIONS,
    DEFAULT_AIM_REPO,
    DEFAULT_ARTIFACTS_FOLDER,
)
from tcbench.modeling import utils, run_contrastive_learning_and_finetune
from tcbench.libtcdatasets import datasets_utils



def main(args):
    """Entry point"""
    args.contrastive_learning_seeds = list(
        map(int, args.contrastive_learning_seeds.split(","))
    )
    args.finetune_seeds = list(map(int, args.finetune_seeds.split(",")))
    args.augmentations = args.augmentations.split(",")
    args.flowpic_dims = list(map(int, args.flowpic_dims.split(",")))
    args.projection_layer_dims = list(map(int, args.projection_layer_dims.split(",")))
    args.dataset = DATASETS.UCDAVISICDM19
    args.dropout = args.dropout.split(",")
    args.suppress_dropout = list(item == "disabled" for item in args.dropout)

    if args.split_indexes is not None:
        args.split_indexes = list(map(int, args.split_indexes.split(",")))

    for aug_name in args.augmentations:
        if aug_name not in DEFAULT_CAMPAIGN_CONTRALEARNANDFINETUNE_VALID_AUGMENTATIONS:
            raise RuntimeError(
                f"Invalid augmentation {aug_name}. Possible choices: {list(DEFAULT_CAMPAIGN_CONTRALEARNANDFINETUNE_VALID_AUGMENTATIONS)}"
            )

    for dim in args.flowpic_dims:
        if dim not in DEFAULT_CAMPAIGN_CONTRALEARNANDFINETUNE_FLOWPICDIMS:
            raise RuntimeError(
                f"Flowpic can only be of size {DEFAULT_CAMPAIGN_CONTRALEARNANDFINETUNE_FLOWPICDIMS}"
            )

    if args.split_indexes is None:
        # NOTE: min_pkts=-1 ...because as is it enforces ucdavis-icdm19
        split_indexes = datasets_utils.get_split_indexes(args.dataset, min_pkts=-1)
    else:
        split_indexes = args.split_indexes

    if args.max_train_splits == -1:
        args.max_train_splits = len(split_indexes)
    split_indexes = split_indexes[: min(len(split_indexes), args.max_train_splits)]

    campaign_id = args.campaign_id
    if campaign_id is None:
        campaign_id = datetime.now().strftime("%s")

    extra_aim_hparams = dict(
        campaign_id=campaign_id,
    )

    experiments_grid = list(
        itertools.product(
            split_indexes,
            args.contrastive_learning_seeds,
            args.finetune_seeds,
            args.flowpic_dims,
            args.suppress_dropout,
            args.projection_layer_dims,
        )
    )

    if len(experiments_grid) == 0:
        raise RuntimeError(f"Something wrong: The experiments grid is empty")

    cum_run_completion_time = 0
    avg_run_completion_time = 0
    for exp_idx, (
        split_index,
        contrastive_learning_seed,
        finetune_seed,
        flowpic_dim,
        suppress_dropout,
        projection_layer_dim,
    ) in enumerate(experiments_grid):
        time_run_start = time.time()

        time_to_completion = timedelta(
            seconds=avg_run_completion_time * (len(experiments_grid) - exp_idx)
        )
        print()
        print("#" * 10)
        print(
            f"# campaign_id: {campaign_id} | run {exp_idx+1}/{len(experiments_grid)} - time to completion {time_to_completion}"
        )
        print("#" * 10)
        print()

        if args.dry_run:
            print(f"split_indexes              ({len(split_indexes)}): {split_indexes}")
            print(
                f"contrastive learning seeds ({len(args.contrastive_learning_seeds)}): {args.contrastive_learning_seeds}"
            )
            print(
                f"finetune seeds             ({len(args.finetune_seeds)}): {args.finetune_seeds}"
            )
            print(
                f"projection layer dims      ({len(args.projection_layer_dims)}): {args.projection_layer_dims}"
            )
            print(f"dropout                    ({len(args.dropout)}): {args.dropout}")
            print(
                f"flowpic dims               ({len(args.flowpic_dims)}): {args.flowpic_dims}"
            )
            sys.exit(0)

        new_params = dict(
            split_index=split_index,
            contrastive_learning_seed=contrastive_learning_seed,
            finetune_seed=finetune_seed,
        )

        # creating a "dummy" Namespace with all
        # default values which will be overwritten
        # based on the campain parameters
        # cmd = f'--config {args.config_fname}'.split()
        cmd = ""
        run_args = run_contrastive_learning_and_finetune.cli_parser().parse_args(cmd)
        extra_aim_hparams["campaign_exp_idx"] = exp_idx

        for attr_name, _ in vars(run_args).items():
            if attr_name in new_params:
                setattr(run_args, attr_name, new_params[attr_name])
        run_args.final = False
        run_args.method = "simclr"
        # run_args.config = args.config
        run_args.aim_experiment_name = args.aim_experiment_name
        run_args.aim_repo = args.aim_repo
        run_args.artifacts_folder = args.artifacts_folder
        run_args.gpu_index = args.gpu_index
        run_args.suppress_dropout = suppress_dropout
        run_args.flowpic_dim = flowpic_dim
        run_args.projection_layer_dim = projection_layer_dim
        # run_args.finetune_augmentation = args.finetune_augmentation
        run_args.augmentations = args.augmentations
        run_args.batch_size = args.batch_size
        run_args.train_val_split_ratio = args.train_val_split_ratio

        run_contrastive_learning_and_finetune.main(run_args, extra_aim_hparams)

        time_run_end = time.time()
        cum_run_completion_time += time_run_end - time_run_start
        avg_run_completion_time = cum_run_completion_time / (exp_idx + 1)


def cli_parser():
    """Create an ArgumentParser"""
    parser = argparse.ArgumentParser()
    ###################
    ## general options
    ###################
    parser.add_argument(
        "--campaign-id",
        default=None,
        help=utils.compose_cli_help_string("A campaign id to mark all experiments"),
    )
    parser.add_argument(
        "--aim-repo",
        default=DEFAULT_AIM_REPO, 
        type=pathlib.Path,
        help=utils.compose_cli_help_string(
            "Local aim folder or URL of AIM remote server"
        ),
    )
    parser.add_argument(
        "--aim-experiment-name",
        default="contrastive-learning-and-finetune",
        help=utils.compose_cli_help_string(
            "The experiment name to use for all Aim run in the campaign"
        ),
    )
    parser.add_argument(
        "--artifacts-folder",
        type=pathlib.Path,
        default=DEFAULT_ARTIFACTS_FOLDER, 
        help=utils.compose_cli_help_string("Artifact folder"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=50,
        help=utils.compose_cli_help_string(
            "Number of parallel worker for loading the data"
        ),
    )
    parser.add_argument(
        "--gpu-index",
        default="0",
        help=utils.compose_cli_help_string("GPU where to operate"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=utils.compose_cli_help_string(
            "Show the number of experiments and then quit"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help=utils.compose_cli_help_string("Training batch size"),
    )
    parser.add_argument(
        "--split-indexes",
        default=None,
        help=utils.compose_cli_help_string(
            "Coma separted list of split indexes. Use -1 to disable predefined split"
        ),
    )

    ###################
    ## data options
    ###################
    parser.add_argument(
        "--max-train-splits",
        type=int,
        default=-1,
        help=utils.compose_cli_help_string(
            "The maximum number of training splits to experiment with. If -1, use all available"
        ),
    )
    parser.add_argument(
        "--augmentations",
        default="changertt,timeshift",
        help=utils.compose_cli_help_string(
            "A pair of augmentations to use for contrastive learning"
        ),
    )
    parser.add_argument(
        "--train-val-split-ratio",
        default=0.8,
        type=float,
        help=utils.compose_cli_help_string(
            "Fraction of samples to dedicate for training"
        )
    )

    ###################
    ## flowpic options
    ###################
    parser.add_argument(
        "--flowpic-dims",
        default=",".join(
            map(str, DEFAULT_CAMPAIGN_CONTRALEARNANDFINETUNE_FLOWPICDIMS)
        ),  # FLOWPIC_DIMS)),
        help=utils.compose_cli_help_string(
            "Coma separated list of flowpic dimensions for experiments"
        ),
    )
    ###################
    ## training options
    ###################
    parser.add_argument(
        "--contrastive-learning-seeds",
        default=",".join(
            map(
                str,
                DEFAULT_CAMPAING_CONTRALEARNANDFINETUNE_SEEDS_CONTRALEARN,
            )
        ),  # SEEDS_CONTRASTIVELEARNING)),
        help=utils.compose_cli_help_string(
            "Coma separated list of seeds to use for contrastive learning pretraining"
        ),
    )
    parser.add_argument(
        "--finetune-seeds",
        default=",".join(
            map(str, DEFAULT_CAMPAIGN_CONTRALEARNANDFINETUNE_SEEDS_FINETUNE)
        ),  ##SEEDS_FINETUNING)),
        help=utils.compose_cli_help_string(
            "Coma separated list of seeds to use for finetune training"
        ),
    )
    parser.add_argument(
        "--dropout",
        type=str,
        default="disabled",
        help=utils.compose_cli_help_string(
            "Coma separated list. Choices: (enabled, disabled)"
        ),
    )

    parser.add_argument(
        "--projection-layer-dims",
        default="30",
        help=utils.compose_cli_help_string(
            "Coma separated list of contrastive learning projection layer dimensions"
        ),
    )
    #    parser.add_argument(
    #        "--finetune-augmentation", default="none",
    #        choices=("none", "only-views", "views-and-original"),
    #        help=utils.compose_cli_help_string("Optional augmentation for finetuning training data. With 'only-views' finetuning is performed only using augmented data; with 'views-and-original' finetuning is performed using augmentation and original data. By default, no augmentation is performed")
    #    )
    return parser


if __name__ == "__main__":
    args = cli_parser().parse_args()
    main(args)
