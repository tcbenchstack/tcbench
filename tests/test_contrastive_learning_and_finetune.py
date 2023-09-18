import pandas as pd

import pytest
import torch
import pathlib

import tcbench
from tcbench.modeling import utils, run_contrastive_learning_and_finetune


@pytest.mark.parametrize(
    "params, expected_artifacts_folder",
    [
        (
            dict(
                dataset_name=tcbench.DATASETS.UCDAVISICDM19,
                learning_rate=0.001,
                batch_size=32,
                flowpic_dim=32,
                split_idx=0,
                seed=12345,
                loss_temperature=0.07,
                with_dropout=False,
                projection_layer_dim=30,
            ),
            pytest.DIR_RESOURCES
            / pathlib.Path(
                "_reference_aim_run/ucdavis-icdm19/simclr-dropout-and-projection/9e2dc14286ab452f992e5c2d"
            ),
        ),
    ],
)
def test_pretrain(tmp_path, params, expected_artifacts_folder):
    params["artifacts_folder"] = tmp_path
    utils.seed_everything(params.get("seed", 12345))

    state = run_contrastive_learning_and_finetune.pretrain(**params)

    # verifying trained model weights
    fname = f'best_model_weights_pretrain_split_{params["split_idx"]}.pt'
    pytest.helpers.verify_deeplearning_model(
        tmp_path / fname, expected_artifacts_folder / fname
    )



@pytest.mark.parametrize(
    "params, expected_artifacts_folder",
    [
        (
            [
                "--dataset ucdavis-icdm19",
                "--contrastive-learning-seed 12345",
                "--finetune-seed 12345",
                "--batch-size 32",
                "--flowpic-dim 32",
                "--split-index 0",
                "--suppress-dropout",
                "--projection-layer-dim 30",
            ],
            pytest.DIR_RESOURCES
            / pathlib.Path(
                "_reference_aim_run/ucdavis-icdm19/simclr-dropout-and-projection/9e2dc14286ab452f992e5c2d"
            ),
        ),
    ],
)
def test_main(tmp_path, params, expected_artifacts_folder):
    params.append(f"--artifacts-folder {tmp_path}/artifacts")

    parser = run_contrastive_learning_and_finetune.cli_parser()
    args = parser.parse_args(" ".join(params).split())
    args.method = "simclr"
    args.augmentations = args.augmentations.split(",")

    run_contrastive_learning_and_finetune.main(args)

    # artifacts are stored into a doubly nested folder
    # as <dataset>/<aim-hash>
    artifacts_folder = next((tmp_path / 'artifacts').iterdir())

    fname_models = sorted(path.name for path in artifacts_folder.glob("*.pt"))
    expected_fname_models = sorted(
        path.name for path in expected_artifacts_folder.glob("*.pt")
    )
    assert fname_models == expected_fname_models

    for fname in fname_models:
        pytest.helpers.verify_deeplearning_model(
            artifacts_folder / fname, expected_artifacts_folder / fname
        )

    # verifying reports
    # note: by using tmp_path / test*.csv automatically
    # skips leftover if suppressed with the command line option
    pytest.helpers.verify_reports(
        artifacts_folder,
        expected_artifacts_folder,
        with_train=False,
        with_val=False,
        with_test=True,
    )
