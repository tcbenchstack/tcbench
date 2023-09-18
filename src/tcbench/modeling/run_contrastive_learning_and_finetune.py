from __future__ import annotations

"""
This is script reproduces the scenarios
from Figure 1, Figure 2 and Figure 3
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
import functools
import sys
import logging

from typing import Dict, Any

import tcbench
from tcbench import (
    DATASETS,
    DEFAULT_ARTIFACTS_FOLDER,
    DEFAULT_AIM_REPO,
    MODELING_DATASET_TYPE,
)
from tcbench.modeling import dataprep, backbone, methods, utils, aimutils


def pretrain(
    dataset_name: str = DATASETS.UCDAVISICDM19,
    dataset_minpkts: int = -1,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    flowpic_dim: int = 32,
    flowpic_block_duration: int = 15,
    split_idx: int = 0,
    aug_samples: int = 2,
    aug_config: dict = None,
    logger=None,
    device: str = "cuda:0",
    tracker: aim.Run = None,
    workers: int = 50,
    artifacts_folder=None,
    seed: int = 12345,
    epochs: int = 50,
    patience_steps: int = 3,
    loss_temperature: float = 0.07,
    with_dropout: bool = True,
    projection_layer_dim: int = 30,
    max_samples_per_class: int = -1,
    state: dict = None,
    train_val_split_ratio: float = 0.8,
):
    """Pretrain a model"""
    assert aug_samples >= 2, "aug_samples cannot be smaller than 2"

    if state is None:
        state = dict()

    utils.seed_everything(seed)

    if aug_config is None:
        aug_config = dict(changertt={}, timeshift={})

    dset_train, dset_val = dataprep.load_dataset(
        dataset_name=dataset_name,
        dataset_type=MODELING_DATASET_TYPE.TRAIN_VAL,
        split_idx=split_idx,
        flowpic_dim=flowpic_dim,
        flowpic_block_duration=flowpic_block_duration,
        aug_config=aug_config,
        aug_samples=aug_samples,
        aug_when_loading=False,
        n_workers=workers,
        logger=logger,
        dataset_minpkts=dataset_minpkts,
        max_samples_per_class=max_samples_per_class,
        seed=seed,
        train_val_split_ratio=train_val_split_ratio,
    )

    train_loader = torch.utils.data.DataLoader(
        dset_train,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(dset_val, batch_size=batch_size)

    net = backbone.net_factory(
        num_classes=None,
        flowpic_dim=32,
        with_dropout=with_dropout,
        projection_layer_dim=projection_layer_dim,
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    trainer = methods.SimCLRTrainer(
        pretrain_config=dict(
            optimizer=optimizer,
            loss_temperature=loss_temperature,
        ),
        deterministic=True,
        device=device,
        tracker=tracker,
        logger=logger,
    )
    best_net = trainer.pretrain_loop(
        net=net,
        train_loader=train_loader,
        val_loader=val_loader,
        patience_monitor=methods.PatienceMonitorAccuracy(
            "acc_top_5", steps=patience_steps
        ),
        epochs=epochs,
        context="contrastivelearning",
    )
    if artifacts_folder:
        fname = artifacts_folder / f"best_model_weights_pretrain_split_{split_idx}.pt"
        utils.log_msg(f"saving: {fname}", logger)
        best_net.save_weights(fname)

    state["best_net"] = best_net
    state["dset_train"] = dset_train
    state["dset_val"] = dset_val

    return state


def finetune_test(
    net: backbone.BaseNet,
    dset: dataprep.FlowpicDataset,
    dset_name: str,
    trainer: methods.SimCLRTrainer,
    logger: logging.Logger = None,
    batch_size=32,
    device: str = "cuda:0",
    tracker: aim.Run = None,
    artifacts_folder=None,
):
    """Test after finetune"""
    utils.log_msg(f"\n--- finetune (test) on {dset_name} ---", logger)
    utils.log_msg(dset.samples_count(), logger)

    loader = torch.utils.data.DataLoader(dset, batch_size, shuffle=False)
    context = f"test-{dset_name}"
    metrics, reports = trainer.finetune_test_loop(
        loader, with_reports=True, context=context
    )

    utils.log_msg(
        f'Test dataset {dset_name} | loss: {metrics["loss"]:.6f} | acc: {metrics["acc"]:.1f}',
        logger,
    )
    utils.classification_reports(
        net,
        dset,
        batch_size,
        device=device,
        context=context,
        save_to=artifacts_folder,
        logger=logger,
    )
    return metrics, reports


def finetune(
    # config,
    dataset_name: str = tcbench.DATASETS.UCDAVISICDM19,
    dataset_minpkts: int = -1,
    batch_size:int=32,
    flowpic_dim:int=32,
    flowpic_block_duration:int=15,
    learning_rate:float=0.01,
    tracker:aim.Run=None,
    logger:logging.Logger=None,
    seed:int=12345,
    epochs:int=50,
    device:str="cuda:0",
    split_idx:int=None,
    artifacts_folder:pathlib.Path=None,
    fname_pretrain_weights:pathlib.Path=None,
    train_samples:int=10,
    patience_steps:int=5,
    patience_min_delta:float=0.001,
    with_dropout: bool = True,
    projection_layer_dim: int = 30,
    aug_config: dict = None,
    aug_samples: int = 2,
    aug_yield_also_original: bool = False,
    state: dict = None,
) -> Dict[str, Any]:
    """Finetune a model"""
    utils.seed_everything(seed)

    if (artifacts_folder is None and fname_pretrain_weights is None) or (
        artifacts_folder is not None and fname_pretrain_weights is not None
    ):
        raise RuntimeError("Provide either artifact_folder or fname_weights")

    if artifacts_folder and split_idx is None:
        raise RuntimeError("When using artifact_folder, split_idx cannot be None")

    fname_pretrain_weights = (
        artifacts_folder / f"best_model_weights_pretrain_split_{split_idx}.pt"
    )

    dset_dict = dataprep.load_dataset(
        dataset_name=dataset_name,
        dataset_type=MODELING_DATASET_TYPE.FINETUNING,
        flowpic_dim=flowpic_dim,
        flowpic_block_duration=flowpic_block_duration,
        logger=logger,
        train_samples=train_samples,
        seed=seed,
        aug_config=aug_config,
        aug_samples=aug_samples,
        aug_yield_also_original=aug_yield_also_original,
        dataset_minpkts=dataset_minpkts,
    )

    dataset_names = [name.split("_")[0] for name in dset_dict if name.endswith("train")]

    if state is None:
        state = dict()

    for dset_name in dataset_names:
        dset_train = dset_dict[f"{dset_name}_train"]
        dset_test = dset_dict[f"{dset_name}_test"]
        state[f"{dset_name}_dset_train"] = dset_train
        state[f"{dset_name}_dset_test"] = dset_test
        num_classes = dset_train.num_classes

        train_loader = torch.utils.data.DataLoader(
            dset_train, batch_size=batch_size, shuffle=True
        )

        utils.log_msg(f"\n--- finetune (train) on {dset_name} ---", logger)

        utils.log_msg(dset_train.samples_count(), logger)

        # the network here is just a dummy object
        # the actual network is created by the trainer
        net = backbone.net_factory(
            num_classes=None,
            flowpic_dim=flowpic_dim,
            with_dropout=with_dropout,
            projection_layer_dim=projection_layer_dim,
        )
        # the optimizer here is just a dummy
        # reference object. The final optimizer
        # is recreated when triggering the training
        # to adapt to the network and loaded weights
        finetune_config = dict(
            optimizer=torch.optim.Adam(net.parameters(), learning_rate),
        )

        trainer = methods.SimCLRTrainer(
            finetune_config=finetune_config,
            deterministic=True,
            device=device,
            tracker=tracker,
            logger=logger,
        )
        best_net = trainer.finetune_loop(
            net=net,
            train_loader=train_loader,
            val_loader=None,
            epochs=epochs,
            num_classes=dset_train.num_classes,
            fname_pretrain_weights=fname_pretrain_weights,
            patience_monitor=methods.PatienceMonitorLoss(
                steps=patience_steps, min_delta=patience_min_delta
            ),
            context=f"finetune_{dset_name}",
        )
        if artifacts_folder:
            fname = (
                artifacts_folder
                / f"best_model_weights_finetune_{dset_name}_from_split_{split_idx}.pt"
            )
            utils.log_msg(f"saving: {fname}", logger)
            best_net.save_weights(fname)

        state[f"{dset_name}_best_net"] = best_net

        metrics, reports = finetune_test(
            net=best_net,
            dset=dset_test,
            dset_name=dset_name,
            trainer=trainer,
            logger=logger,
            batch_size=batch_size,
            device=device,
            tracker=tracker,
            artifacts_folder=artifacts_folder,
        )
        state[f"{dset_name}_class_rep"] = reports["class_rep"]
        state[f"{dset_name}_conf_mtx"] = reports["conf_mtx"]

    return state


def main(args, extra_aim_hparams=None):
    """Entry point"""
    # bounding to a specific gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    args.method = "simclr"

    if extra_aim_hparams is None:
        extra_aim_hparams = {}

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
    # artifacts_folder = args.artifacts_folder / args.dataset / aim_run_hash
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
        split_index=args.split_index,
        dataset=args.dataset,
        dataset_minpkts=args.dataset_minpkts,
        contrastive_learning_seed=args.contrastive_learning_seed,
        finetune_seed=args.finetune_seed,
        finetune_train_samples=args.finetune_train_samples,
        with_dropout=with_dropout,
        projection_layer_dim=args.projection_layer_dim,
        finetune_augmentation=args.finetune_augmentation,
        augmentations=args.augmentations,
        train_val_split_ratio=args.train_val_split_ratio,
        **extra_aim_hparams,
    )
    aim_run["hparams"] = run_hparams

    utils.log_msg("--- run hparams ---", logger)
    for param_name, param_value in run_hparams.items():
        utils.log_msg(f"{param_name}: {param_value}", logger)
    utils.log_msg("-------------------", logger)

    generic_params = dict(
        dataset_name=args.dataset,
        dataset_minpkts=args.dataset_minpkts,
        tracker=aim_run,
        logger=logger,
        artifacts_folder=artifacts_folder,
        with_dropout=with_dropout,
        projection_layer_dim=args.projection_layer_dim,
    )
    flowpic_params = dict(
        flowpic_dim=args.flowpic_dim,
        flowpic_block_duration=args.flowpic_block_duration,
    )

    aug_config = dict(changertt={}, timeshift={})
    if args.augmentations:
        aug_config = {aug_name: {} for aug_name in args.augmentations}

    contrastive_learning_params = dict(
        split_idx=args.split_index,
        batch_size=args.batch_size,
        learning_rate=args.contrastive_learning_lr,
        seed=args.contrastive_learning_seed,
        aug_config=aug_config,
        aug_samples=2,
        epochs=args.contrastive_learning_epochs,
        patience_steps=args.contrastive_learning_patience_steps,
        loss_temperature=args.contrastive_learning_temperature,
        max_samples_per_class=args.max_samples_per_class,
        train_val_split_ratio=args.train_val_split_ratio,
    )
    state_pretrain = pretrain(
        **generic_params, **flowpic_params, **contrastive_learning_params
    )

    finetune_params = dict(
        batch_size=args.batch_size,
        learning_rate=args.finetune_lr,
        seed=args.finetune_seed,
        epochs=args.finetune_epochs,
        split_idx=args.split_index,
        train_samples=args.finetune_train_samples,
        patience_steps=args.finetune_patience_steps,
        patience_min_delta=args.finetune_patience_min_delta,
    )
    if args.finetune_augmentation != "none":
        finetune_params["aug_config"] = contrastive_learning_params["aug_config"]
        finetune_params["aug_samples"] = contrastive_learning_params["aug_samples"]
        finetune_params["aug_yield_also_original"] = (
            args.finetune_augmentation == "views-and-original"
        )
    state_finetune = finetune(**generic_params, **flowpic_params, **finetune_params)

    state = dict()
    for key, value in state_pretrain.items():
        state[f"pretrain_{key}"] = value
    for key, valu in state_finetune.items():
        state[f"finetune_{key}"] = value

    aim_run.close()

    utils.dump_cli_args(args, artifacts_folder / "params.yml", logger=logger)
    return state


def cli_parser():
    """Create an ArgumentParser"""
    parser = argparse.ArgumentParser()

    ####################################
    # general configs
    ####################################
    # parser.add_argument(
    #    "--config", "-c", type=pathlib.Path, required=True, default="./config.yml",
    #    help=utils.compose_cli_help_string("General configuration file"),
    # )
    parser.add_argument(
        "--artifacts-folder",
        type=pathlib.Path,
        default="./debug/artifacts",
        help=utils.compose_cli_help_string("Artifact folder"),
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help=utils.compose_cli_help_string("Device where to run experiments"),
    )
    parser.add_argument(
        "--aim-repo",
        default="./debug",
        help=utils.compose_cli_help_string(
            "Local aim folder or URL of AIM remote server"
        ),
    )
    parser.add_argument(
        "--aim-experiment-name",
        default="contrastive-learning-and-finetune",
        help=utils.compose_cli_help_string(
            "The experiment name to use for the Aim run"
        ),
    )
    parser.add_argument(
        "--gpu-index",
        default="0",
        help=utils.compose_cli_help_string("The GPU id to use"),
    )
    parser.add_argument("--final", action="store_true", default=False)

    ####################################
    # data and dataset configs
    ####################################
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
        "--workers",
        type=int,
        default=50,
        help=utils.compose_cli_help_string(
            "Number of parallel worker for loading the data"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help=utils.compose_cli_help_string("Training batch size"),
    )
    parser.add_argument(
        "--split-index",
        type=int,
        default=0,
        help=utils.compose_cli_help_string("Datasplit index"),
    )
    parser.add_argument(
        "--augmentations",
        default="changertt,timeshift",
        help=utils.compose_cli_help_string(
            "A pair of augmentations to use for contrastive learning"
        ),
    )
    parser.add_argument(
        "--max-samples-per-class",
        default=-1,
        type=int,
        help=utils.compose_cli_help_string(
            "Balance the dataset with the specified number of samples per class"
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

    ####################################
    # flowpic configs
    ####################################
    parser.add_argument(
        "--flowpic-dim",
        type=int,
        default=32,
        choices=(32,),
        help=utils.compose_cli_help_string("Flowpic dimension"),
    )
    parser.add_argument(
        "--flowpic-block-duration",
        type=int,
        default=15,
        help=utils.compose_cli_help_string("Time window from which extract a flowpic"),
    )

    ####################################
    # model architecture
    ####################################
    parser.add_argument(
        "--suppress-dropout",
        default=False,
        action="store_true",
        help=utils.compose_cli_help_string("Mask dropout layers with Identity"),
    )
    parser.add_argument(
        "--finetune-augmentation",
        default="none",
        choices=("none", "only-views", "views-and-original"),
        help=utils.compose_cli_help_string(
            "Optional augmentation for finetuning training data. With 'only-views' finetuning is performed only using augmented data; with 'views-and-original' finetuning is performed using augmentation and original data. By default, no augmentation is performed"
        ),
    )
    parser.add_argument(
        "--projection-layer-dim",
        default=30,
        type=int,
        help=utils.compose_cli_help_string(
            "The number of units in the contrastive learning projection layer"
        ),
    )

    ####################################
    # contrastive learning configs
    ####################################
    parser.add_argument(
        "--contrastive-learning-lr",
        type=float,
        default=0.001,
        help=utils.compose_cli_help_string("Learning rate for pretraining"),
    )
    parser.add_argument(
        "--contrastive-learning-seed",
        type=int,
        default=12345,
        help=utils.compose_cli_help_string("Seed for contrastive learning pretraining"),
    )
    parser.add_argument(
        "--contrastive-learning-patience-steps",
        type=int,
        default=3,
        help=utils.compose_cli_help_string(
            "Max steps to wait before stopping training if the top5 validation accuracy does not improve"
        ),
    )
    parser.add_argument(
        "--contrastive-learning-temperature",
        type=float,
        default=0.07,
        help=utils.compose_cli_help_string("Temperature for InfoNCE loss"),
    )
    parser.add_argument(
        "--contrastive-learning-epochs",
        type=int,
        default=50,
        help=utils.compose_cli_help_string(
            "Epochs for contrastive learning pretraining"
        ),
    )

    ####################################
    # finetune configs
    ####################################
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=0.01,
        help=utils.compose_cli_help_string("Learning for for finetune"),
    )
    parser.add_argument(
        "--finetune-patience-steps",
        type=int,
        default=5,
        help=utils.compose_cli_help_string(
            "Max steps to wait before stopping training training loss does not improve"
        ),
    )
    parser.add_argument(
        "--finetune-patience-min-delta",
        type=float,
        default=0.001,
        help=utils.compose_cli_help_string("Min improvement for training loss"),
    )
    parser.add_argument(
        "--finetune-train-samples",
        type=int,
        default=10,
        help=utils.compose_cli_help_string(
            "Number of samples per-class for finetune training"
        ),
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=50,
        help=utils.compose_cli_help_string("Epochs for finetune training"),
    )
    parser.add_argument(
        "--finetune-seed",
        type=int,
        default=12345,
        help=utils.compose_cli_help_string("Sed for finetune training"),
    )

    return parser


if __name__ == "__main__":
    parser = cli_parser()

    args = parser.parse_args()

    args.augmentations = args.augmentations.split(",")

    main(args)
