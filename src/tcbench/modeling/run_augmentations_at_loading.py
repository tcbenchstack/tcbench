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
import sys

import tcbench
from tcbench import DATASETS, DEFAULT_ARTIFACTS_FOLDER, DEFAULT_AIM_REPO
from tcbench.modeling import (
    dataprep,
    backbone,
    methods,
    utils,
    aimutils,
    MODELING_DATASET_TYPE,
)


def train(
    dataset_name: DATASETS = tcbench.DATASETS.UCDAVISICDM19,
    dataset_minpkts: int = -1,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    flowpic_dim: int = 32,
    flowpic_block_duration: int = 15,
    split_index: int = 0,
    max_samples_per_class: int = -1,
    logger=None,
    aug_name: str = "noaug",
    aug_samples: int = 10,
    device: str = "cuda:0",
    tracker: aim.Run = None,
    workers: int = 50,
    artifacts_folder=None,
    seed: int = 12345,
    epochs: int = 50,
    patience_steps: int = 5,
    patience_min_delta: float = 0.001,
    train_val_split_ratio: float = 0.8,
    suppress_val_augmentation: bool = False,
    with_dropout: bool = True,
    state: dict = None,
) -> Dict[str, Any]:
    """Model training"""
    if state is None:
        state = dict()

    aug_config = {aug_name: dict()}

    dset_train, dset_val = dataprep.load_dataset(
        dataset_name=dataset_name,
        dataset_type=MODELING_DATASET_TYPE.TRAIN_VAL,
        split_idx=split_index,
        max_samples_per_class=max_samples_per_class,
        flowpic_dim=flowpic_dim,
        flowpic_block_duration=flowpic_block_duration,
        aug_config=aug_config,
        aug_samples=aug_samples,
        aug_when_loading=True,
        n_workers=workers,
        suppress_val_augmentation=suppress_val_augmentation,
        logger=logger,
        seed=seed,
        dataset_minpkts=dataset_minpkts,
    )
    train_loader = torch.utils.data.DataLoader(dset_train, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dset_val, batch_size, shuffle=False)

    net = backbone.net_factory(
        num_classes=dset_train.num_classes,
        flowpic_dim=flowpic_dim,
        with_dropout=with_dropout,
    )

    #torchsummary.summary(net.to(device), (1, flowpic_dim, flowpic_dim))
    utils.log_msg("\nnetwork architecture", logger)
    utils.log_torchsummary(net.to(device), (1, flowpic_dim, flowpic_dim), logger)
    optimizer = torch.optim.Adam(net.parameters(), learning_rate)

    trainer_kwargs = dict(
        net=net, optimizer=optimizer, tracker=tracker, device=device, logger=logger
    )
    trainer = methods.trainer_factory("monolithic", **trainer_kwargs)

    best_net = trainer.train_loop(
        epochs=epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        patience_monitor=methods.PatienceMonitorLoss(
            steps=patience_steps, min_delta=patience_min_delta
        ),
    )

    state = dict(
        best_net=best_net,
        dset_train=dset_train,
        dset_val=dset_val,
    )

    if artifacts_folder is not None:
        best_net.save_weights(
            artifacts_folder / f"best_model_weights_split_{split_index}.pt"
        )

    reports = utils.classification_reports(
        best_net,
        dset_train,
        batch_size,
        device,
        context="train",
        save_to=artifacts_folder,
        logger=logger,
    )
    state["train_class_rep"] = reports["class_rep"]
    state["train_conf_mtx"] = reports["conf_mtx"]

    report = utils.classification_reports(
        best_net,
        dset_val,
        batch_size,
        device,
        context="val",
        save_to=artifacts_folder,
        logger=logger,
    )
    state["val_class_rep"] = report["class_rep"]
    state["val_conf_mtx"] = report["conf_mtx"]

    return state


def test(
    dataset_name: tcbench.DATASETS.UCDAVISICDM19,
    dataset_minpkts: int = -1,
    split_idx: int = 0,
    batch_size: int = 32,
    flowpic_dim: int = 32,
    flowpic_block_duration: int = 15,
    logger=None,
    device: str = "cuda:0",
    tracker: aim.Run = None,
    artifacts_folder: pathlib.Path = None,
    with_dropout: bool = True,
    state: dict = None,
):
    """Model testing"""
    if state is None:
        state = dict()

    dset_dict = dataprep.load_dataset(
        dataset_name=dataset_name,
        dataset_type=MODELING_DATASET_TYPE.TEST,
        flowpic_dim=flowpic_dim,
        flowpic_block_duration=flowpic_block_duration,
        logger=logger,
        dataset_minpkts=dataset_minpkts,
    )

    # pick the first dataset name
    # just to identify the number of classes
    name = next(iter(dset_dict.keys()))
    num_classes = dset_dict[name].num_classes

    net = backbone.net_factory(
        num_classes=num_classes, flowpic_dim=flowpic_dim, with_dropout=with_dropout
    )
    fname = artifacts_folder / f"./best_model_weights_split_{split_idx}.pt"
    net.load_weights(fname)
    net = net.to(device)
    trainer = methods.trainer_factory(
        method="monolithic", net=net, device=device, tracker=tracker, logger=logger
    )

    for name, dset in dset_dict.items():
        loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False)
        context = "test"
        if name != "test":
            context = f"test-{name}"
        metrics, reports = trainer.test_loop(loader, with_reports=True, context=context)

        utils.log_msg(
            f'Test dataset {name} | loss: {metrics["loss"]:.6f} | acc: {metrics["acc"]:.1f}',
            logger,
        )
        reports = utils.classification_reports(
            net,
            dset,
            batch_size,
            device=device,
            context=context,
            save_to=artifacts_folder,
            logger=logger,
        )
        state[f"{name}_class_rep"] = reports["class_rep"]
        state[f"{name}_conf_mtx"] = reports["conf_mtx"]

        class_rep = reports["class_rep"]
        precision, recall, f1 = class_rep.loc[
            "weighted avg", ["precision", "recall", "f1-score"]
        ].values
        aimutils.track_metrics(
            tracker, dict(precision=precision, recall=recall, f1=f1), context=context
        )

    state.update(dset_dict)
    return state


def test_with_train_val_leftover(
    dataset_name: tcbench.DATASETS.UCDAVISICDM19,
    dset_train: dataprep.FlowpicDataset,
    dset_val: dataprep.FlowpicDataset,
    split_idx: int,
    batch_size: int = 32,
    flowpic_dim: int = 32,
    flowpic_block_duration: int = 15,
    logger=None,
    device: str = "cuda:0",
    tracker: aim.Run = None,
    artifacts_folder: pathlib.Path = None,
    with_dropout: bool = True,
    state: dict = None,
    dataset_minpkts: int = -1,
):
    """Model testing on leftover split"""
    if state is None:
        state = dict()

    dset_leftover = dataprep.load_dataset(
        dataset_name=dataset_name,
        dataset_type=MODELING_DATASET_TYPE.TRAIN_VAL_LEFTOVER,
        dset_train=dset_train,
        dset_val=dset_val,
        flowpic_dim=flowpic_dim,
        logger=logger,
        dataset_minpkts=dataset_minpkts,
    )

    num_classes = dset_train.num_classes

    net = backbone.net_factory(
        num_classes=num_classes, flowpic_dim=flowpic_dim, with_dropout=with_dropout
    )
    fname = artifacts_folder / f"./best_model_weights_split_{split_idx}.pt"
    net.load_weights(fname)
    net = net.to(device)
    trainer = methods.trainer_factory(
        method="monolithic", net=net, device=device, tracker=tracker, logger=logger
    )

    loader = torch.utils.data.DataLoader(
        dset_leftover, batch_size=batch_size, shuffle=False
    )
    context = f"test-train-val-leftover"
    metrics, reports = trainer.test_loop(loader, with_reports=True, context=context)
    utils.log_msg(
        f'Test dataset train-val-leftover | loss: {metrics["loss"]:.6f} | acc: {metrics["acc"]:.1f}',
        logger,
    )
    reports = utils.classification_reports(
        net,
        dset_leftover,
        batch_size,
        device=device,
        context=context,
        save_to=artifacts_folder,
        logger=logger,
    )
    state["dset_leftover"] = dset_leftover
    state["leftover_class_rep"] = reports["class_rep"]
    state["leftover_conf_mtx"] = reports["conf_mtx"]

    class_rep = reports["class_rep"]
    precision, recall, f1 = class_rep.loc[
        "weighted avg", ["precision", "recall", "f1-score"]
    ].values
    aimutils.track_metrics(
        tracker, dict(precision=precision, recall=recall, f1=f1), context=context
    )

    return state


def main(args, extra_aim_hparams=None) -> Dict[str, Any]:
    """Entry point"""
    # bounding to a specific gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    args.method = "monolithic"

    if extra_aim_hparams is None:
        extra_aim_hparams = {}

    utils.seed_everything(args.seed)

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
            f"WARNING: the artifact folder is not a subfolder of the AIM repo",
            logger
        )

    with_dropout = not args.suppress_dropout
    run_hparams = dict(
        flowpic_dim=args.flowpic_dim,
        flowpic_block_duration=args.flowpic_block_duration,
        split_index=args.split_index,
        max_samples_per_class=args.max_samples_per_class,
        aug_name=args.aug_name,
        patience_steps=args.patience_steps,
        suppress_val_augmentation=args.suppress_val_augmentation,
        dataset=args.dataset,
        dataset_minpkts=args.dataset_minpkts,
        seed=args.seed,
        with_dropout=with_dropout,
        **extra_aim_hparams,
    )
    aim_run["hparams"] = run_hparams

    utils.log_msg("--- run hparams ---", logger)
    for param_name, param_value in run_hparams.items():
        utils.log_msg(f"{param_name}: {param_value}", logger)
    utils.log_msg("-------------------", logger)

    state = dict()

    state = train(
        dataset_name=args.dataset,
        dataset_minpkts=args.dataset_minpkts,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience_steps=args.patience_steps,
        flowpic_dim=args.flowpic_dim,
        flowpic_block_duration=args.flowpic_block_duration,
        split_index=args.split_index,
        max_samples_per_class=args.max_samples_per_class,
        aug_name=args.aug_name,
        tracker=aim_run,
        workers=args.workers,
        artifacts_folder=artifacts_folder,
        logger=logger,
        seed=args.seed,
        epochs=args.epochs,
        train_val_split_ratio=args.train_val_split_ratio,
        suppress_val_augmentation=args.suppress_val_augmentation,
        with_dropout=with_dropout,
        state=state,
    )
    state = test(
        dataset_name=args.dataset,
        dataset_minpkts=args.dataset_minpkts,
        split_idx=args.split_index,
        batch_size=args.batch_size,
        flowpic_dim=args.flowpic_dim,
        flowpic_block_duration=args.flowpic_block_duration,
        tracker=aim_run,
        artifacts_folder=artifacts_folder,
        logger=logger,
        with_dropout=with_dropout,
        state=state,
    )

    if not args.suppress_test_train_val_leftover and args.dataset == "ucdavis-icdm19":
        state = test_with_train_val_leftover(
            dataset_name=args.dataset,
            dset_train=state["dset_train"],
            dset_val=state["dset_val"],
            split_idx=args.split_index,
            batch_size=args.batch_size,
            flowpic_dim=args.flowpic_dim,
            flowpic_block_duration=args.flowpic_block_duration,
            tracker=aim_run,
            artifacts_folder=artifacts_folder,
            logger=logger,
            with_dropout=with_dropout,
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
    #    parser.add_argument(
    #        "--config",
    #        "-c",
    #        type=pathlib.Path,
    #        required=True,
    #        default="./config.yml",
    #        help=utils.compose_cli_help_string("General configuration file"),
    #    )
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
        "--gpu-index",
        default="0",
        help=utils.compose_cli_help_string("The GPU id to use"),
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
        default="augmentation-at-loading",
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
        choices=tuple(map(str, tcbench.DATASETS.__members__.values())),
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
    parser.add_argument(
        "--split-index",
        type=int,
        default=0,
        help=utils.compose_cli_help_string("Datasplit index"),
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=-1,
        help=utils.compose_cli_help_string(
            "Activated when --split-index is -1 to define how many samples to select for train+val (with a 80/20 split between train and val"
        ),
    )

    ###############
    # training
    ###############
    parser.add_argument(
        "--train-val-split-ratio",
        type=float,
        default=0.8,
        help=utils.compose_cli_help_string("Training train/val split"),
    )
    parser.add_argument(
        "--aug-name",
        type=str,
        choices=(
            "noaug",
            "rotate",
            "horizontalflip",
            "colorjitter",
            "packetloss",
            "timeshift",
            "changertt",
        ),
        default="noaug",
        help=utils.compose_cli_help_string("Augmentation policy"),
    )
    parser.add_argument(
        "--suppress-val-augmentation",
        action="store_true",
        default=False,
        help=utils.compose_cli_help_string("Do not augment validation set"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help=utils.compose_cli_help_string("Random seed"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help=utils.compose_cli_help_string("Training batch size"),
    )
    parser.add_argument("--patience-steps", default=5, type=int)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help=utils.compose_cli_help_string("Traning learning rate"),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help=utils.compose_cli_help_string("Number of epochs for training"),
    )
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
    return parser


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    parser = cli_parser()

    args = parser.parse_args()

    # args.config = utils.load_config(args.config)
    # args.method = "monolithic"

    main(args)
