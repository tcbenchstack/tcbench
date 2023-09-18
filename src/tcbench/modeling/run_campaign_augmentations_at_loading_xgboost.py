import numpy as np
import pandas as pd

from datetime import datetime, timedelta

import itertools
import argparse
import pathlib
import torch
import time
import sys

from tcbench.modeling import utils, run_augmentations_at_loading_xgboost
from tcbench.libtcdatasets import datasets_utils

AUGMENTATIONS = ["noaug"]

from tcbench import (
    DEFAULT_CAMPAIGN_AUGATLOAD_SEEDS,
    DEFAULT_CAMPAIGN_AUGATLOAD_PKTSERIESLEN,
    DEFAULT_CAMPAIGN_AUGATLOAD_FLOWPICDIMS,
    DATASETS,
    DEFAULT_AIM_REPO,
    DEFAULT_ARTIFACTS_FOLDER,
)

def main(args):
    """Entry point"""

    args.seeds = list(map(int, args.seeds.split(",")))
    args.max_n_pkts = list(map(int, args.max_n_pkts.split(",")))
    args.flowpic_dims = list(map(int, args.flowpic_dims.split(",")))

    # forcing specific parameters
    args.dataset = DATASETS.UCDAVISICDM19
    args.augmentations = AUGMENTATIONS
    args.dataset_minpkts = -1

    if args.split_indexes is not None:
        args.split_indexes = list(map(int, args.split_indexes.split(",")))

    for dim in args.flowpic_dims:
        if (
            dim not in DEFAULT_CAMPAIGN_AUGATLOAD_FLOWPICDIMS
        ):  # FLOWPIC_DIMS: #(32, 64, 1500):
            raise RuntimeError(
                f"Invalid value {dim}: Flowpic can only be of size {DEFAULT_CAMPAIGN_AUGATLOAD_FLOWPICDIMS}"
            )

    if args.split_indexes is None:
        split_indexes = datasets_utils.get_split_indexes(
            args.dataset, args.dataset_minpkts
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

    l = [split_indexes, args.augmentations, args.seeds]
    # force a dummy value which will be ignored
    # in the downstream task
    if args.flow_representation == "flowpic":
        l.append(args.flowpic_dims)
        l.append([30])
    else:
        l.append([32])
        l.append(args.max_n_pkts)

    experiments_grid = list(itertools.product(*l))

    cum_run_completion_time = 0
    avg_run_completion_time = 0
    for exp_idx, (split_index, aug_name, seed, flowpic_dim, mnp) in enumerate(
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
            print(f"seeds         ({len(args.seeds)}): {args.seeds}")
            if args.flow_representation == "flowpic":
                print(f"flowpic_dims  ({len(args.flowpic_dims)}): {args.flowpic_dims}")
            else:
                print(f"max_n_pkts    ({len(args.max_n_pkts)}): {args.max_n_pkts}")
            sys.exit(0)

        new_params = dict(
            split_index=split_index,
            aug_name=aug_name,
            flowpic_dim=flowpic_dim,
            max_n_pkts=mnp,
            seed=seed,
        )

        extra_aim_hparams["campaign_exp_idx"] = exp_idx + 1

        # creating a "dummy" Namespace with all
        # default values which will be overwritten
        # based on the campain parameters
        # cmd = f'--config {args.config_fname}'.split()
        cmd = ""
        run_args = run_augmentations_at_loading_xgboost.cli_parser().parse_args(cmd)
        for attr_name, _ in vars(run_args).items():
            if attr_name in new_params:
                setattr(run_args, attr_name, new_params[attr_name])

        # directly passing parameters
        # run_args.final = False
        run_args.method = "xgboost"
        run_args.aim_experiment_name = args.aim_experiment_name
        # run_args.gpu_index = args.gpu_index
        run_args.aim_repo = args.aim_repo
        run_args.artifacts_folder = args.artifacts_folder
        # run_args.patience_steps = args.patience_steps
        run_args.suppress_test_train_val_leftover = (
            args.suppress_test_train_val_leftover
        )
        run_args.max_samples_per_class = args.max_samples_per_class
        # run_args.suppress_val_augmentation = args.suppress_val_augmentation
        # run_args.suppress_dropout = args.suppress_dropout
        run_args.dataset = args.dataset
        run_args.flow_representation = args.flow_representation
        run_args.batch_size = 32

        run_augmentations_at_loading_xgboost.main(run_args, extra_aim_hparams)

        time_run_end = time.time()
        cum_run_completion_time += time_run_end - time_run_start
        avg_run_completion_time = cum_run_completion_time / (exp_idx + 1)


def cli_parser():
    """Create an ArgumentParser"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aim-repo",
        default=DEFAULT_AIM_REPO,
        help=utils.compose_cli_help_string(
            "Local aim folder or URL of AIM remote server"
        ),
    )
    parser.add_argument(
        "--artifacts-folder",
        type=pathlib.Path,
        default=DEFAULT_ARTIFACTS_FOLDER,
        help=utils.compose_cli_help_string("Artifact folder"),
    )
    parser.add_argument(
        "--campaign-id",
        default=None,
        help=utils.compose_cli_help_string("A campaign id to mark all experiments"),
    )
    parser.add_argument(
        "--split-indexes",
        default=None,
        help=utils.compose_cli_help_string(
            "Comma separted list of split indexes. Use -1 to disable predefined split"
        ),
    )
    parser.add_argument(
        "--max-train-splits",
        type=int,
        default=-1,
        help=utils.compose_cli_help_string(
            "The maximum number of training splits to experiment with. If -1, use all available"
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
        "--seeds",
        default=",".join(
            map(
                str,
                DEFAULT_CAMPAIGN_AUGATLOAD_SEEDS,
            )
        ),  # SEEDS)),
        help=utils.compose_cli_help_string(
            "Coma separated list of seed for experiments"
        ),
    )
    parser.add_argument(
        "--aim-experiment-name",
        default="xgb_pkts",
        help=utils.compose_cli_help_string(
            "The experiment name to use for all Aim run in the campaign"
        ),
    )
    parser.add_argument(
        "--flow-representation",
        choices=("flowpic", "pktseries"),
        default="flowpic",
        help=utils.compose_cli_help_string(
            "The string representing the flow representation (flowpic or pktseries)"
        ),
    )
    parser.add_argument(
        "--max-n-pkts",
        default=",".join(
            map(
                str,
                DEFAULT_CAMPAIGN_AUGATLOAD_PKTSERIESLEN,
            )
        ), 
        help=utils.compose_cli_help_string(
            "The number of packets in case of xgboost on packet series"
        ),
    )
    parser.add_argument(
        "--suppress-test-train-val-leftover",
        default=False,
        action="store_true",
        help=utils.compose_cli_help_string("Skip test on leftover split"),
    )
    parser.add_argument(
        "--flowpic-dims",
        default=",".join(
            map(str, DEFAULT_CAMPAIGN_AUGATLOAD_FLOWPICDIMS)
        ),
        help=utils.compose_cli_help_string(
            "Coma separated list of flowpic dimensions for experiments"
        ),
    )
    #    parser.add_argument(
    #        "--dataset",
    #        choices=(str(DATASETS.UCDAVISICDM19),),#('ucdavis-icdm19', #)'utmobilenet21'),
    #        default=str(DATASETS.UCDAVISICDM19), #'ucdavis-icdm19',
    #        help=utils.compose_cli_help_string("Dataset to use for modeling"),
    #    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=utils.compose_cli_help_string(
            "Show the number of experiments and then quit"
        ),
    )

    return parser


if __name__ == "__main__":
    args = cli_parser().parse_args()

    main(args)
