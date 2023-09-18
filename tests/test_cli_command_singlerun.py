import pytest
import pathlib
import re
import shlex

from click.testing import CliRunner

from tcbench.cli.main import main
import tcbench


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
                "--no-test-leftover",
                "--method monolithic",
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
                "--method monolithic",
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
                #                '--no-test-leftover',
                "--no-dropout",
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
                #                '--no-test-leftover',
                "--no-dropout",
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
                #                '--no-test-leftover',
                "--no-dropout",
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
                #                '--no-test-leftover',
                "--no-dropout",
            ],
            pytest.DIR_RESOURCES
            / pathlib.Path(
                "_reference_aim_run/mirage22/augmentation-at-loading-no-dropout/minpkts1000/7f8d861e91ee442eaddd6802"
            ),
        ),
       # ...using a larger dataset
        (
            [
                f"--dataset {tcbench.DATASETS.UCDAVISICDM19}",
                "--batch-size 32",
                "--flowpic-dim 32",
                "--split-index -1",
                "--seed 6",
                "--aug-name noaug",
                "--no-dropout",
                "--no-test-leftover",
            ],
            pytest.DIR_RESOURCES
            / pathlib.Path(
                "_reference_aim_run/ucdavis-icdm19/larger-trainset/augmentation-at-loading/artifacts/ca30dd098e5146738adf30a5"
            ),
        ),
        (
            [
                f"--dataset {tcbench.DATASETS.UCDAVISICDM19}",
                "--batch-size 32",
                "--flowpic-dim 32",
                "--split-index -1",
                "--seed 18",
                "--aug-name noaug",
                "--no-dropout",
                "--no-test-leftover",
            ],
            pytest.DIR_RESOURCES
            / pathlib.Path(
                "_reference_aim_run/ucdavis-icdm19/larger-trainset/augmentation-at-loading/artifacts/79eefb2dc25e42d8992c757f"
            ),
        ),
        ##############
        # xgboost
        ##############
        (
            [
                f"--dataset {str(tcbench.DATASETS.UCDAVISICDM19)}",
                "--input-repr pktseries",
                "--pktseries-len 10",
                "--split-index 0",
                "--seed 12345",
                "--method xgboost",
                "--no-test-leftover",
            ],
            pytest.DIR_RESOURCES
            / pathlib.Path(
                "_reference_aim_run/ucdavis-icdm19/xgboost/noaugmentation-timeseries/5fa59c129a3e4aa6bb9b7640"
            ),
        ),
    ],
)
def test_augment_at_load(tmp_path, cli_params, expected_artifacts_folder):
    text_cmd = " ".join(
        [
            "run",
            "augment-at-loading",
            f"--artifacts-folder {tmp_path}",
        ]
        + cli_params
    )

    res = re.findall(r"--split-index (-?\d+)", text_cmd)
    if not res:
        raise RuntimeError("missing --split-index as input parameter")
    split_index = res[0]

    clirunner = CliRunner()

    result = clirunner.invoke(main, shlex.split(text_cmd), catch_exceptions=False)

    # the artifact folder depends on the aim run hash
    artifacts_folder = next(tmp_path.iterdir())

    if "xgboost" not in text_cmd:
        fname = f"best_model_weights_split_{split_index}.pt"
        ref_fname = fname
        epsilon = None
        if split_index == '-1':
            epsilon = 10e-10
            ref_fname = next(expected_artifacts_folder.glob('best_model_weights_split*'))
        pytest.helpers.verify_deeplearning_model(
            artifacts_folder / fname, 
            expected_artifacts_folder / ref_fname, 
            epsilon
        )
    else:
        fname = f"xgb_model_split_{split_index}.json"
        #pytest.helpers.verify_md5_model(
        #    artifacts_folder / fname, expected_artifacts_folder / fname
        #)

    pytest.helpers.verify_reports(artifacts_folder, expected_artifacts_folder)


@pytest.mark.parametrize(
    "cli_params, expected_artifacts_folder",
    [
        (
            [
                f"--dataset {str(tcbench.DATASETS.UCDAVISICDM19)}",
                "--batch-size 32",
                "--flowpic-dim 32",
                "--split-index 0",
                "--no-dropout",
                "--cl-seed 12345",
                "--ft-seed 12345",
                "--cl-projection-layer-dim 30",
            ],
            pytest.DIR_RESOURCES
            / pathlib.Path(
                "_reference_aim_run/ucdavis-icdm19/simclr-dropout-and-projection/9e2dc14286ab452f992e5c2d"
            ),
        ),
        # ...using larger dataset
        (
            [
                f"--dataset {str(tcbench.DATASETS.UCDAVISICDM19)}",
                "--batch-size 32",
                "--flowpic-dim 32",
                "--split-index -1",
                "--no-dropout",
                "--cl-seed 32",
                "--ft-seed 2",
                "--cl-projection-layer-dim 30",
            ],
            pytest.DIR_RESOURCES
            / pathlib.Path(
                "_reference_aim_run/ucdavis-icdm19/larger-trainset/simclr/0cd7d318591a4c3ead8f63e8"
            ),
        ),
        (
            [
                f"--dataset {str(tcbench.DATASETS.UCDAVISICDM19)}",
                "--batch-size 32",
                "--flowpic-dim 32",
                "--split-index -1",
                "--no-dropout",
                "--cl-seed 18",
                "--ft-seed 18",
                "--cl-projection-layer-dim 30",
            ],
            pytest.DIR_RESOURCES
            / pathlib.Path(
                "_reference_aim_run/ucdavis-icdm19/larger-trainset/simclr/2e197d7c78774db18f4faa6b"
            ),
        ),
    ],
)
def test_contralearn_and_finetune(tmp_path, cli_params, expected_artifacts_folder):
    text_cmd = " ".join(
        [
            "run",
            "contralearn-and-finetune",
            f"--artifacts-folder {tmp_path}/artifacts",
        ]
        + cli_params
    )

    res = re.findall(r"--split-index (-?\d+)", text_cmd)
    if not res:
        raise RuntimeError("missing --split-index as input parameter")
    split_index = res[0]

    clirunner = CliRunner()

    result = clirunner.invoke(main, shlex.split(text_cmd), catch_exceptions=False)

    # artifacts are stored into a doubly nested folder
    # as <dataset>/<aim-hash>
    artifacts_folder = next((tmp_path / 'artifacts').iterdir())

    fname_models = sorted(path.name for path in artifacts_folder.glob("*.pt"))
    expected_fname_models = sorted(
        path.name for path in expected_artifacts_folder.glob("*.pt")
    )
    assert len(fname_models) == len(expected_fname_models)

    for fname in fname_models:
        epsilon = None
        ref_fname = fname
        if "--split-index -1" in text_cmd:
            epsilon=10e-12
            ref_fname = next(expected_artifacts_folder.glob(f'{fname.split("split_")[0]}*pt'))
        pytest.helpers.verify_deeplearning_model(
            artifacts_folder / fname, expected_artifacts_folder / ref_fname, epsilon
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
