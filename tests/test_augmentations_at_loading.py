# import pandas as pd

import pytest

# import torch
import pathlib

import tcbench
from tcbench.modeling import utils, run_augmentations_at_loading, MODELING_DATASET_TYPE, aimutils


@pytest.mark.parametrize(
    "params, expected_artifacts_folder",
    [
        (
            dict(
                dataset_name=tcbench.DATASETS.UCDAVISICDM19,
                learning_rate=0.001,
                batch_size=32,
                flowpic_dim=32,
                split_index=0,
                seed=12345,
                aug_name="noaug",
            ),
            pytest.DIR_RESOURCES
            / pathlib.Path(
                "_reference_aim_run/ucdavis-icdm19/augmentation-at-loading-with-dropout/a2e34aee53144add903a6d66/"
            ),
        ),
        (
            dict(
                dataset_name=tcbench.DATASETS.UTMOBILENET21,
                dataset_minpkts=10,
                learning_rate=0.001,
                batch_size=32,
                flowpic_dim=32,
                split_index=0,
                seed=12345,
                aug_name="noaug",
                with_dropout=False,
            ),
            pytest.DIR_RESOURCES
            / pathlib.Path(
                "_reference_aim_run/utmobilenet21/augmentation-at-loading-no-dropout/minpkts10/29e039f661e74b9599f1738c"
            ),
        ),
        (
            dict(
                dataset_name=tcbench.DATASETS.MIRAGE19,
                dataset_minpkts=10,
                learning_rate=0.001,
                batch_size=32,
                flowpic_dim=32,
                split_index=0,
                seed=42,
                aug_name="noaug",
                with_dropout=False,
            ),
            pytest.DIR_RESOURCES
            / pathlib.Path(
                "_reference_aim_run/mirage19/augmentation-at-loading-no-dropout/minpkts10/12a2fa303fc842d99c07b03d"
            ),
        ),
        (
            dict(
                dataset_name=tcbench.DATASETS.MIRAGE22,
                dataset_minpkts=10,
                learning_rate=0.001,
                batch_size=32,
                flowpic_dim=32,
                split_index=0,
                seed=12345,
                aug_name="noaug",
                with_dropout=False,
            ),
            pytest.DIR_RESOURCES
            / pathlib.Path(
                "_reference_aim_run/mirage22/augmentation-at-loading-no-dropout/minpkts10/63c06bffb6a447a78cd5c670"
            ),
        ),
        (
            dict(
                dataset_name=tcbench.DATASETS.MIRAGE22,
                dataset_minpkts=1000,
                learning_rate=0.001,
                batch_size=32,
                flowpic_dim=32,
                split_index=0,
                seed=12345,
                aug_name="noaug",
                with_dropout=False,
            ),
            pytest.DIR_RESOURCES
            / pathlib.Path(
                "_reference_aim_run/mirage22/augmentation-at-loading-no-dropout/minpkts1000/7f8d861e91ee442eaddd6802"
            ),
        ),
    ],
)
def test_train(tmp_path, params, expected_artifacts_folder):
    params["artifacts_folder"] = tmp_path
    utils.seed_everything(params.get("seed", 12345))

    state = run_augmentations_at_loading.train(**params)

    # verifying trained model weights
    fname = f'best_model_weights_split_{params["split_index"]}.pt'
    pytest.helpers.verify_deeplearning_model(
        tmp_path / fname, expected_artifacts_folder / fname
    )
    pytest.helpers.verify_reports(
        tmp_path,
        expected_artifacts_folder,
        with_train=True,
        with_val=True,
        with_test=False,
    )


@pytest.mark.parametrize(
    "cli_params, expected_artifacts_folder",
    [
        (
            [
                f"--dataset {tcbench.DATASETS.UCDAVISICDM19}",
                "--learning-rate 0.001",
                "--batch-size 32",
                "--flowpic-dim 32",
                "--split-index 0",
                "--seed 12345",
                "--aug-name noaug",
                "--suppress-test-train-val-leftover",
            ],
            pytest.DIR_RESOURCES
            / pathlib.Path(
                "_reference_aim_run/ucdavis-icdm19/augmentation-at-loading-with-dropout/a2e34aee53144add903a6d66"
            ),
        ),
        # same as previous but testing also leftover
        (
            [
                f"--dataset {tcbench.DATASETS.UCDAVISICDM19}",
                "--learning-rate 0.001",
                "--batch-size 32",
                "--flowpic-dim 32",
                "--split-index 0",
                "--seed 12345",
                "--aug-name noaug",
            ],
            pytest.DIR_RESOURCES
            / pathlib.Path(
                "_reference_aim_run/ucdavis-icdm19/augmentation-at-loading-with-dropout/a2e34aee53144add903a6d66"
            ),
        ),
        (
            [
                f"--dataset {tcbench.DATASETS.UTMOBILENET21}",
                "--dataset-minpkts 10",
                "--learning-rate 0.001",
                "--batch-size 32",
                "--flowpic-dim 32",
                "--split-index 0",
                "--seed 12345",
                "--aug-name noaug",
                "--suppress-test-train-val-leftover",
                "--suppress-dropout",
            ],
            pytest.DIR_RESOURCES
            / pathlib.Path(
                "_reference_aim_run/utmobilenet21/augmentation-at-loading-no-dropout/minpkts10/29e039f661e74b9599f1738c"
            ),
        ),
        (
            [
                f"--dataset {tcbench.DATASETS.MIRAGE19}",
                "--dataset-minpkts 10",
                "--learning-rate 0.001",
                "--batch-size 32",
                "--flowpic-dim 32",
                "--split-index 0",
                "--seed 42",
                "--aug-name noaug",
                "--suppress-test-train-val-leftover",
                "--suppress-dropout",
            ],
            pytest.DIR_RESOURCES
            / pathlib.Path(
                "_reference_aim_run/mirage19/augmentation-at-loading-no-dropout/minpkts10/12a2fa303fc842d99c07b03d"
            ),
        ),
        (
            [
                f"--dataset {tcbench.DATASETS.MIRAGE22}",
                "--dataset-minpkts 10",
                "--learning-rate 0.001",
                "--batch-size 32",
                "--flowpic-dim 32",
                "--split-index 0",
                "--seed 12345",
                "--aug-name noaug",
                "--suppress-test-train-val-leftover",
                "--suppress-dropout",
            ],
            pytest.DIR_RESOURCES
            / pathlib.Path(
                "_reference_aim_run/mirage22/augmentation-at-loading-no-dropout/minpkts10/63c06bffb6a447a78cd5c670"
            ),
        ),
        (
            [
                f"--dataset {tcbench.DATASETS.MIRAGE22}",
                "--dataset-minpkts 1000",
                "--learning-rate 0.001",
                "--batch-size 32",
                "--flowpic-dim 32",
                "--split-index 0",
                "--seed 12345",
                "--aug-name noaug",
                "--suppress-test-train-val-leftover",
                "--suppress-dropout",
            ],
            pytest.DIR_RESOURCES
            / pathlib.Path(
                "_reference_aim_run/mirage22/augmentation-at-loading-no-dropout/minpkts1000/7f8d861e91ee442eaddd6802"
            ),
        ),
    ],
)
def test_main(tmp_path, cli_params, expected_artifacts_folder):
    cli_params.append(f"--artifacts-folder {tmp_path}/artifacts")
    cli_params.append(f'--aim-repo {tmp_path}')

    parser = run_augmentations_at_loading.cli_parser()
    cmd = (" ".join(cli_params)).split()
    args = parser.parse_args(cmd)
    args.method = "monolithic"

    state = run_augmentations_at_loading.main(args)

    # the artifact folder depends on the aim run hash
    tmp_path = next((tmp_path / 'artifacts').iterdir())

    fname = f"best_model_weights_split_{args.split_index}.pt"
    pytest.helpers.verify_deeplearning_model(
        tmp_path / fname, expected_artifacts_folder / fname
    )
    pytest.helpers.verify_reports(tmp_path, expected_artifacts_folder)
