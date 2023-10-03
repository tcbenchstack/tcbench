import numpy as np
import pandas as pd

from datetime import datetime, timedelta

import itertools
import argparse
import pathlib
import torch
import time
import sys

import tcbench
from tcbench import (
    DATASETS,
    DEFAULT_CAMPAIGN_AUGATLOAD_AUGMENTATIONS,
    DEFAULT_CAMPAIGN_AUGATLOAD_SEEDS,
    DEFAULT_CAMPAIGN_AUGATLOAD_FLOWPICDIMS,
)
from tcbench.modeling import utils, run_augmentations_at_loading
from tcbench.libtcdatasets import datasets_utils

def main(args):
    """Entry point"""
    torch.multiprocessing.set_start_method("spawn", force=True)

    args.seeds = list(map(int, args.seeds.split(",")))
    args.flowpic_dims = list(map(int, args.flowpic_dims.split(",")))
    args.augmentations = list(args.augmentations.split(","))

    if args.split_indexes is not None:
        args.split_indexes = list(map(int, args.split_indexes.split(",")))

    for dim in args.flowpic_dims:
        if dim not in DEFAULT_CAMPAIGN_AUGATLOAD_FLOWPICDIMS: 
            raise RuntimeError(
                f"Flowpic can only be of size {DEFAULT_CAMPAIGN_AUGATLOAD_FLOWPICDIMS}"
            )

    for aug_name in args.augmentations:
        if aug_name not in DEFAULT_CAMPAIGN_AUGATLOAD_AUGMENTATIONS:
            raise RuntimeError(
                f"Invalid augmentation {arg_name}. Possible choices: {DEFAULT_CAMPAIGN_AUGATLOAD_AUGMENTATIONS}"
            )

    dataset_name = args.dataset

    if args.split_indexes is None:
        split_indexes = datasets_utils.get_split_indexes(
            dataset_name, args.dataset_minpkts
        )
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
            split_indexes, args.augmentations, args.flowpic_dims, args.seeds
        )
    )
    cum_run_completion_time = 0
    avg_run_completion_time = 0
    for exp_idx, (split_index, aug_name, flowpic_dim, seed) in enumerate(
        experiments_grid
    ):
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
            print(f"split_indexes ({len(split_indexes)}): {split_indexes}")
            print(f"augmentations ({len(args.augmentations)}): {args.augmentations}")
            print(f"flowpic_dims  ({len(args.flowpic_dims)}): {args.flowpic_dims}")
            print(f"seeds         ({len(args.seeds)}): {args.seeds}")
            sys.exit(0)

        new_params = dict(
            split_index=split_index,
            aug_name=aug_name,
            flowpic_dim=flowpic_dim,
            seed=seed,
        )

        extra_aim_hparams["campaign_exp_idx"] = exp_idx + 1

        # creating a "dummy" Namespace with all
        # default values which will be overwritten
        # based on the campain parameters
        # cmd = f'--config {args.config_fname}'.split()
        cmd = ""
        run_args = run_augmentations_at_loading.cli_parser().parse_args(cmd)
        for attr_name, _ in vars(run_args).items():
            if attr_name in new_params:
                setattr(run_args, attr_name, new_params[attr_name])

        # directly passing parameters
        run_args.final = False
        run_args.method = "monolithic"
        run_args.experiment_name = args.aim_experiment_name
        run_args.gpu_index = args.gpu_index
        run_args.aim_repo = args.aim_repo
        run_args.artifacts_folder = args.artifacts_folder
        run_args.patience_steps = args.patience_steps
        run_args.suppress_test_train_val_leftover = (
            args.suppress_test_train_val_leftover
        )
        run_args.max_samples_per_class = args.max_samples_per_class
        # run_args.suppress_val_augmentation = args.suppress_val_augmentation
        run_args.suppress_dropout = args.suppress_dropout
        run_args.dataset = args.dataset
        run_args.dataset_minpkts = args.dataset_minpkts
        run_args.epochs = args.epochs
        run_args.batch_size = args.batch_size

        run_augmentations_at_loading.main(run_args, extra_aim_hparams)

        time_run_end = time.time()
        cum_run_completion_time += time_run_end - time_run_start
        avg_run_completion_time = cum_run_completion_time / (exp_idx + 1)


def cli_parser():
    """Create an ArgumentParser"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--aim-repo",
        default="./debug",
        help=utils.compose_cli_help_string(
            "Local aim folder or URL of AIM remote server"
        ),
    )
    parser.add_argument(
        "--artifacts-folder",
        type=pathlib.Path,
        default="./debug/artifacts",
        help=utils.compose_cli_help_string("Artifact folder"),
    )
    parser.add_argument(
        "--campaign-id",
        default=None,
        help=utils.compose_cli_help_string("A campaign id to mark all experiments"),
    )
    parser.add_argument(
        "--aim-experiment-name",
        default="augmentations-at-loading",
        help=utils.compose_cli_help_string(
            "The experiment name to use for all Aim run in the campaign"
        ),
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

    #############################
    # data options
    #############################
    parser.add_argument(
        "--split-indexes",
        default=None,
        help=utils.compose_cli_help_string(
            "Coma separted list of split indexes. Use -1 to disable predefined split"
        ),
    )
    parser.add_argument(
        "--max-samples-per-class",
        default=-1,
        type=int,
        help=utils.compose_cli_help_string(
            "Used in conjuction with --split-indexes -1 to dynamically generate a train/val split. The number of samples specified corresponds to train+val (which will be separated with 80/20 for train and val)"
        ),
    )
    parser.add_argument(
        "--augmentations",
        default=",".join(
            map(str, DEFAULT_CAMPAIGN_AUGATLOAD_AUGMENTATIONS)
        ),  # AUGMENTATIONS)),
        help=utils.compose_cli_help_string(
            "Coma separated list of augmentations for experiments"
        ),
    )
    #    parser.add_argument(
    #        "--dataset",
    #        choices=('ucdavis-icdm19', 'utmobilenet21', 'mirage19', 'mirage22'),
    #        default='ucdavis-icdm19',
    #        help=utils.compose_cli_help_string("Dataset to use for modeling"),
    #    )
    parser.add_argument(
        "--dataset",
        choices=tuple(map(str, DATASETS.__members__.values())),
        default=str(tcbench.DATASETS.UCDAVISICDM19),
        help=utils.compose_cli_help_string("Dataset to use for modeling"),
    )
    parser.add_argument(
        "--dataset-minpkts",
        choices=(-1, 10, 100, 1000),
        default=-1,
        type=int,
        help=utils.compose_cli_help_string(
            "When used in combination with --dataset can refine the dataset and split to use for modeling"
        ),
    )

    #############################
    # train options
    #############################
    parser.add_argument(
        "--max-train-splits",
        type=int,
        default=-1,
        help=utils.compose_cli_help_string(
            "The maximum number of training splits to experiment with. If -1, use all available"
        ),
    )
    parser.add_argument(
        "--seeds",
        default=",".join(map(str, DEFAULT_CAMPAIGN_AUGATLOAD_SEEDS)),  # SEEDS)),
        help=utils.compose_cli_help_string(
            "Coma separated list of seed for experiments"
        ),
    )

    parser.add_argument(
        "--flowpic-dims",
        default=",".join(
            map(str, DEFAULT_CAMPAIGN_AUGATLOAD_FLOWPICDIMS)
        ),  # FLOWPIC_DIMS)),
        help=utils.compose_cli_help_string(
            "Coma separated list of flowpic dimensions for experiments"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help=utils.compose_cli_help_string("Batch size"),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help=utils.compose_cli_help_string("Learning rate"),
    )
    parser.add_argument("--patience-steps", default=5, type=int)
    #    parser.add_argument(
    #        "--suppress-val-augmentation",
    #        action='store_true',
    #        default=False,
    #        help=utils.compose_cli_help_string('Do not augment validation set')
    #    )
    parser.add_argument(
        "--suppress-test-train-val-leftover",
        default=False,
        action="store_true",
        help=utils.compose_cli_help_string("Skip test on leftover split"),
    )
    parser.add_argument(
        "--suppress-dropout",
        default=False,
        action="store_true",
        help=utils.compose_cli_help_string("Mask dropout layers with Identity"),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help=utils.compose_cli_help_string("Number of epochs for training"),
    )

    return parser


if __name__ == "__main__":
    #torch.multiprocessing.set_start_method("spawn")

    parser = cli_parser()
    args = parser.parse_args()

    main(args)
