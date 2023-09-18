from __future__ import annotations

"""
This is script reproduces the scenarios
reported in Table 1, Table 2 and Table 3
of the paper "A Few Shots Traffic Classification 
with mini-FlowPic Augmentations", IMC22
https://dl.acm.org/doi/pdf/10.1145/3517745.3561436
"""

import numpy as np

import aim
import torch
import argparse
import pathlib
import torchsummary
import random
import os
import yaml
import torch
import xgboost as xgb

import tcbench
from tcbench import DATASETS, DEFAULT_ARTIFACTS_FOLDER, DEFAULT_AIM_REPO
from tcbench.modeling import (
    dataprep,
    backbone,
    methods,
    utils,
    MODELING_DATASET_TYPE,
    MODELING_INPUT_REPR_TYPE,
    aimutils,
)


def train(
    dataset_name: DATASETS = DATASETS.UCDAVISICDM19,
    flow_representation: MODELING_INPUT_REPR_TYPE = MODELING_INPUT_REPR_TYPE.FLOWPIC,
    max_n_pkts: int = 10,
    batch_size: int = 32,
    flowpic_dim: int = 32,
    flowpic_block_duration: int = 15,
    split_index: int = 0,
    logger=None,
    tracker: aim.Run = None,
    workers: int = 50,
    artifacts_folder=None,
    seed: int = 12345,
    train_val_split_ratio: float = 0.8,
    state: dict = None,
) -> Dict[str, Any]:
    """Train an XGBoost model"""
    if state is None:
        state = dict()

    dset_train, dset_val = dataprep.load_dataset(
        dataset_name=dataset_name,
        dataset_type=MODELING_DATASET_TYPE.TRAIN_VAL,
        flow_representation=flow_representation,
        max_n_pkts=max_n_pkts,
        split_idx=split_index,
        flowpic_dim=flowpic_dim,
        flowpic_block_duration=flowpic_block_duration,
        n_workers=workers,
        logger=logger,
        seed=seed,
    )

    train_loader = torch.utils.data.DataLoader(dset_train, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dset_val, batch_size, shuffle=False)

    xgboost_model = backbone.xgboost_factory(random_state=seed)

    trainer_kwargs = dict(xgboost_model=xgboost_model, tracker=tracker, logger=logger)
    trainer = methods.trainer_factory("xgboost", **trainer_kwargs)

    trained_model = trainer.train_loop(train_loader=train_loader, val_loader=val_loader)

    xgboost_model.save_model(artifacts_folder / f"xgb_model_split_{split_index}.json")

    utils.classification_reports(
        None,
        dset_train,
        batch_size,
        None,
        context="train",
        save_to=artifacts_folder,
        logger=logger,
        method="xgboost",
        xgboost_model=xgboost_model,
    )
    utils.classification_reports(
        None,
        dset_val,
        batch_size,
        None,
        context="val",
        save_to=artifacts_folder,
        logger=logger,
        method="xgboost",
        xgboost_model=xgboost_model,
    )

    state = dict(
        best_net=xgboost_model,
        dset_train=dset_train,
        dset_val=dset_val,
        scaler=dset_train.scaler,
    )
    return state


def test(
    dataset_name: DATASETS = DATASETS.UCDAVISICDM19,
    flow_representation: MODELING_INPUT_REPR_TYPE = MODELING_INPUT_REPR_TYPE.FLOWPIC,
    max_n_pkts: int = 10,
    split_idx: int = 0,
    batch_size: int = 32,
    flowpic_dim: int = 32,
    flowpic_block_duration: int = 15,
    logger=None,
    tracker: aim.Run = None,
    artifacts_folder: pathlib.Path = None,
    state: dict = None,
):
    """Test an XGBoost model"""
    if state is None:
        state = dict()

    dset_dict = dataprep.load_dataset(
        dataset_name=dataset_name,
        dataset_type=MODELING_DATASET_TYPE.TEST,
        flowpic_dim=flowpic_dim,
        flowpic_block_duration=flowpic_block_duration,
        logger=logger,
        flow_representation=flow_representation,
        max_n_pkts=max_n_pkts,
    )

    # pick the first dataset name
    # just to identify the number of classes
    name = next(iter(dset_dict.keys()))
    num_classes = dset_dict[name].num_classes

    fname = artifacts_folder / f"./xgb_model_split_{split_idx}.json"
    model_xgb_2 = xgb.XGBClassifier()
    model_xgb_2.load_model(fname)
    trainer = methods.trainer_factory(
        method="xgboost", xgboost_model=model_xgb_2, tracker=tracker, logger=logger
    )

    for name, dset in dset_dict.items():
        loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False)
        context = "test"
        dset.set_scaler(state["scaler"])
        if name != "test":
            context = f"test-{name}"

        metrics, reports = trainer.test_loop(loader, with_reports=True, context=context)

        utils.log_msg(
            f'Test dataset {name} |  acc: {metrics["acc"]:.1f}',
            logger,
        )
        utils.classification_reports(
            None,
            dset,
            batch_size,
            device="-1",
            context=context,
            save_to=artifacts_folder,
            logger=logger,
            method="xgboost",
            xgboost_model=model_xgb_2,
        )
    state.update(dset_dict)
    return state


def test_with_train_val_leftover(
    dataset_name: DATASETS,
    flow_representation: MODELING_INPUT_REPR_TYPE,
    max_n_pkts: int,
    dset_train: dataprep.FlowpicDataset,
    dset_val: dataprep.FlowpicDataset,
    split_idx: int,
    batch_size: int = 32,
    flowpic_dim: int = 32,
    flowpic_block_duration: int = 15,
    logger=None,
    tracker: aim.Run = None,
    artifacts_folder: pathlib.Path = None,
    state: dict = None,
):
    """Test and XGBoost model on a leftover split"""
    if state is None:
        state = dict()

    dset_leftover = dataprep.load_dataset(
        dataset_name=dataset_name,
        dataset_type=MODELING_DATASET_TYPE.TRAIN_VAL_LEFTOVER,
        dset_train=dset_train,
        dset_val=dset_val,
        flowpic_dim=flowpic_dim,
        logger=logger,
        flow_representation=flow_representation,
        max_n_pkts=max_n_pkts,
    )

    num_classes = dset_train.num_classes

    fname = artifacts_folder / f"./xgb_model_split_{split_idx}.json"
    model_xgb_2 = xgb.XGBClassifier()
    model_xgb_2.load_model(fname)
    trainer = methods.trainer_factory(
        method="xgboost",
        net=None,
        device="-1",
        tracker=tracker,
        logger=logger,
        xgboost_model=model_xgb_2,
    )

    dset_leftover.set_scaler(state["scaler"])
    loader = torch.utils.data.DataLoader(
        dset_leftover, batch_size=batch_size, shuffle=False
    )
    context = f"test-train-val-leftover"

    metrics, reports = trainer.test_loop(loader, with_reports=True, context=context)
    utils.log_msg(
        f'Test dataset train-val-leftover |  acc: {metrics["acc"]:.1f}',
        logger,
    )
    utils.classification_reports(
        None,
        dset_leftover,
        batch_size,
        device="-1",
        context=context,
        save_to=artifacts_folder,
        logger=logger,
        method="xgboost",
        xgboost_model=model_xgb_2,
    )
    state["dset_leftover"] = dset_leftover
    return state


def main(args, extra_aim_hparams=None) -> Dict[str, Any]:
    """Entry point"""
    if extra_aim_hparams is None:
        extra_aim_hparams = {}

    args.method = "xgboost"
    args.dataset = DATASETS.from_str(str(args.dataset))
    args.flow_representation = MODELING_INPUT_REPR_TYPE.from_str(
        str(args.flow_representation)
    )
    args.aug_name = "noaug"

    if str(args.aim_repo).startswith("aim://"):
        utils.log_msg(f"Connecting to remote AIM server {args.aim_repo}")
        aim_repo_path = args.aim_repo
    else:
        aim_repo_path = pathlib.Path(args.aim_repo)
        args.artifacts_folder = pathlib.Path(args.artifacts_folder)

    aimutils.init_repository(aim_repo_path)
    aim_run = aim.Run(
        repo=aim_repo_path,
        experiment=args.aim_experiment_name,
        log_system_params=True,
        capture_terminal_logs=True,
    )
    aim_run_hash = utils.get_aim_run_hash(aim_run)

    artifacts_folder = args.artifacts_folder / aim_run_hash
    logger = utils.get_logger(artifacts_folder / "log.txt")

    utils.log_msg(f"\nconnecting to AIM repo at: {aim_repo_path}", logger)
    utils.log_msg(f"created aim run hash={aim_run_hash}", logger)
    utils.log_msg(f"artifacts folder at: {artifacts_folder}", logger)
    if artifacts_folder.parent != aim_repo_path:
        utils.log_msg(
            f"WARNING: the artifact folder is not a subfolder of the AIM repo"
        )

    run_hparams = dict(
        flowpic_dim=args.flowpic_dim,
        flowpic_block_duration=args.flowpic_block_duration,
        split_index=args.split_index,
        dataset=str(args.dataset),
        seed=args.seed,
        flow_representation=str(args.flow_representation),
        max_n_pkts=args.max_n_pkts,
        **extra_aim_hparams,
    )
    aim_run["hparams"] = run_hparams

    utils.log_msg("--- run hparams ---")
    for param_name, param_value in run_hparams.items():
        utils.log_msg(f"{param_name}: {param_value}")
    utils.log_msg("-------------------")

    state = dict()

    state = train(
        dataset_name=args.dataset,
        flow_representation=args.flow_representation,
        max_n_pkts=args.max_n_pkts,
        batch_size=args.batch_size,
        flowpic_dim=args.flowpic_dim,
        flowpic_block_duration=args.flowpic_block_duration,
        split_index=args.split_index,
        tracker=aim_run,
        workers=args.workers,
        artifacts_folder=artifacts_folder,
        logger=logger,
        seed=args.seed,
        train_val_split_ratio=args.train_val_split_ratio,
        state=state,
    )
    state = test(
        dataset_name=args.dataset,
        flow_representation=args.flow_representation,
        max_n_pkts=args.max_n_pkts,
        split_idx=args.split_index,
        batch_size=args.batch_size,
        flowpic_dim=args.flowpic_dim,
        flowpic_block_duration=args.flowpic_block_duration,
        tracker=aim_run,
        artifacts_folder=artifacts_folder,
        logger=logger,
        state=state,
    )

    if (
        not args.suppress_test_train_val_leftover
        and args.dataset == DATASETS.UCDAVISICDM19
    ):
        state = test_with_train_val_leftover(
            dataset_name=args.dataset,
            flow_representation=args.flow_representation,
            max_n_pkts=args.max_n_pkts,
            dset_train=state["dset_train"],
            dset_val=state["dset_val"],
            split_idx=args.split_index,
            batch_size=args.batch_size,
            flowpic_dim=args.flowpic_dim,
            flowpic_block_duration=args.flowpic_block_duration,
            tracker=aim_run,
            artifacts_folder=artifacts_folder,
            logger=logger,
            state=state,
        )

    aim_run.close()

    utils.dump_cli_args(args, artifacts_folder / "params.yml", logger=logger)
    return state


def cli_parser():
    """Create an ArgumentParser"""
    parser = argparse.ArgumentParser()

    ##################
    # general config
    ##################
    parser.add_argument(
        "--artifacts-folder",
        type=pathlib.Path,
        default=DEFAULT_ARTIFACTS_FOLDER,
        help=utils.compose_cli_help_string("Artifact folder"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help=utils.compose_cli_help_string(
            "Number of parallel worker for loading the data"
        ),
    )
    parser.add_argument(
        "--aim-repo",
        default=DEFAULT_AIM_REPO,
        help=utils.compose_cli_help_string(
            "Local aim folder or URL of AIM remote server"
        ),
    )
    parser.add_argument(
        "--aim-experiment-name",
        default="xgb_pktseries",
        help=utils.compose_cli_help_string(
            "The name of the experiment for AIM tracking"
        ),
    )
    parser.add_argument("--final", action="store_true", default=False)

    ###############
    # flowpic
    ###############
    parser.add_argument(
        "--flowpic-dim",
        type=int,
        choices=(32, 64, 1500),
        default=32,
        help=utils.compose_cli_help_string("Flowpic dimension"),
    )
    parser.add_argument(
        "--flowpic-block-duration",
        type=int,
        default=15,
        help=utils.compose_cli_help_string("Flowpic block duration (in seconds)"),
    )

    ###############
    # data
    ###############
    parser.add_argument(
        "--dataset",
        choices=(str(DATASETS.UCDAVISICDM19),),
        default=str(DATASETS.UCDAVISICDM19),
        help=utils.compose_cli_help_string("Dataset to use for modeling"),
    )
    parser.add_argument(
        "--split-index",
        type=int,
        default=0,
        help=utils.compose_cli_help_string("Datasplit index"),
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
    parser.add_argument(
        "--train-val-split-ratio",
        type=float,
        default=0.8,
        help=utils.compose_cli_help_string("Training train/val split"),
    )
    #    parser.add_argument(
    #        "--aug-name",
    #        type=str,
    #        choices=(
    #            "noaug",
    #            "rotate",
    #            "horizontalflip",
    #            "colorjitter",
    #            "packetloss",
    #            "timeshift",
    #            "changertt",
    #        ),
    #        default="noaug",
    #        help=utils.compose_cli_help_string("Augmentation policy")
    #    )
    #    parser.add_argument(
    #        "--suppress-val-augmentation",
    #        action='store_true',
    #        default=False,
    #        help=utils.compose_cli_help_string('Do not augment validation set')
    #    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help=utils.compose_cli_help_string("Random seed"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help=utils.compose_cli_help_string("Training batch size"),
    )
    parser.add_argument(
        "--suppress-test-train-val-leftover",
        default=False,
        action="store_true",
        help=utils.compose_cli_help_string("Skip test on leftover split"),
    )
    parser.add_argument(
        "--flow-representation",
        choices=tuple(
            map(str, MODELING_INPUT_REPR_TYPE.__members__.values())
        ),  # ('flowpic', 'pktseries'),
        default=str(MODELING_INPUT_REPR_TYPE.FLOWPIC),  # "flowpic",
        help=utils.compose_cli_help_string(
            "The string representing the flow representation"
        ),
    )
    parser.add_argument(
        "--max-n-pkts",
        type=int,
        default=10,
        help=utils.compose_cli_help_string(
            "The number of packets in case of xgboost on packet series"
        ),
    )
    return parser


if __name__ == "__main__":
    parser = cli_parser()

    args = parser.parse_args()

    main(args)
