import pytest
import pathlib
import re
import shlex

from click.testing import CliRunner

from tcbench.cli.main import main
import tcbench


@pytest.mark.parametrize(
    "cli_params, params_to_match, expected_artifacts_folder",
    [
        (
            [ 
                f"--dataset {tcbench.DATASETS.UCDAVISICDM19}",
                "--batch-size 32",
                "--seeds 12345,42",
                "--flowpic-dims 32",
                "--split-indexes 0",
                "--augmentations noaug",
                "--no-test-leftover",
                "--method monolithic",
            ],
            ["batch_size", "seed", "flowpic_dim", "split_index", "aug_name",],
            pytest.DIR_RESOURCES / "_reference_aim_campaign/ucdavis-icdm19/noaug_2seeds/artifacts",
        ),
        (
            [ 
                f"--dataset {tcbench.DATASETS.UCDAVISICDM19}",
                "--seeds 12345,42",
                "--split-indexes 0,1",
                "--method xgboost",
                "--input-repr pktseries",
                "--pktseries-len 10",
                "--no-test-leftover",
            ],
            ["batch_size", "seed", "max_n_pkts", "split_index", "aug_name",],
            pytest.DIR_RESOURCES / "_reference_aim_campaign/ucdavis-icdm19/xgboost_10pkts_2seeds_2splits/artifacts",
        ),
        (
            [ 
                f"--dataset {tcbench.DATASETS.UCDAVISICDM19}",
                "--seeds 12345,42",
                "--split-indexes 0",
                "--method xgboost",
                "--input-repr flowpic",
                "--flowpic-dims 32",
                "--no-test-leftover",
            ],
            ["batch_size", "seed", "flowpic_dim", "split_index", "aug_name",],
            pytest.DIR_RESOURCES / "_reference_aim_campaign/ucdavis-icdm19/xgboost_flowpic_2seeds_2splits/artifacts",
        ),
    ],
)
def test_augment_at_load(tmp_path, cli_params, params_to_match, expected_artifacts_folder):
    text_cmd = " ".join(
        [
            "campaign",
            "augment-at-loading",
            f"--aim-repo {tmp_path}",
            f"--artifacts-folder {tmp_path}/artifacts",
        ]
        + cli_params
    )

    clirunner = CliRunner()
    result = clirunner.invoke(main, shlex.split(text_cmd), catch_exceptions=False)

    # the artifact folder depends on the aim run hash
    artifacts_folder = tmp_path/'artifacts'

    artifacts_found = len(list(artifacts_folder.iterdir()))
    artifacts_expected = len(list(expected_artifacts_folder.iterdir()))

    assert artifacts_found == artifacts_expected

    pairs = pytest.helpers.match_run_hashes(artifacts_folder, expected_artifacts_folder, params_to_match)
    assert len(pairs) == artifacts_expected

    for run_hash, ref_run_hash in pairs:
        assert ref_run_hash is not None

        curr_artifacts_folder = artifacts_folder / run_hash
        curr_ref_artifacts_folder = expected_artifacts_folder / ref_run_hash

        if "xgboost" not in text_cmd:
            fname = next(curr_artifacts_folder.glob(f"best_model_weights_split_*.pt")).name
            ref_fname = fname
            epsilon=None
            if '--split-indexes -1' in text_cmd:
                ref_fname = next(curr_ref_artifacts_folder.glob(f"best_model_weights_split_*.pt")).name
                epsilon=10e-12

            pytest.helpers.verify_deeplearning_model(
                curr_artifacts_folder / fname, 
                curr_ref_artifacts_folder / ref_fname,
                epsilon=epsilon,
            )
#        else:
#            #fname = f"xgb_model_split_{split_index}.json"
#            fname = next(curr_artifacts_folder.glob('xgb_model_split_*.json')).name
#            pytest.helpers.verify_md5_model(
#                curr_artifacts_folder / fname, 
#                curr_ref_artifacts_folder / fname, 
#            )
        pytest.helpers.verify_reports(
            curr_artifacts_folder,
            curr_ref_artifacts_folder, 
        )


@pytest.mark.parametrize(
    "cli_params, params_to_match, expected_artifacts_folder",
    [
        (
            [
                #f"--dataset {str(tcbench.DATASETS.UCDAVISICDM19)}",
                "--batch-size 32",
                "--flowpic-dims 32",
                "--split-indexes 0",
                "--cl-seeds 12345",
                "--ft-seeds 12345,1,2,3,4",
                "--cl-projection-layer-dims 30",
                "--dropout disabled",
            ],
            ["batch_size", "contrastive_learning_seed", "finetune_seed", "flowpic_dim", "split_index"],
            pytest.DIR_RESOURCES
            / pathlib.Path(
                "_reference_aim_campaign/ucdavis-icdm19/simclr_1-5seeds/artifacts/"
            ),
        ),
    ],
)
def test_contralearn_and_finetune(tmp_path, cli_params, params_to_match, expected_artifacts_folder):
    text_cmd = " ".join(
        [
            "campaign",
            "contralearn-and-finetune",
            f"--aim-repo {tmp_path}",
            f"--artifacts-folder {tmp_path}/artifacts",
        ]
        + cli_params
    )

    clirunner = CliRunner()

    result = clirunner.invoke(main, shlex.split(text_cmd), catch_exceptions=False)

    if result.exit_code != 0:
        exc_type, exc_value, exc_traceback = result.exc_info
        print(text_cmd)
        print(result.output)
        raise exc_type(exc_value).with_traceback(exc_traceback)

    # the artifact folder depends on the aim run hash
    artifacts_folder = tmp_path/'artifacts'

    artifacts_found = len(list(artifacts_folder.iterdir()))
    artifacts_expected = len(list(expected_artifacts_folder.iterdir()))

    assert artifacts_found == artifacts_expected

    pairs = pytest.helpers.match_run_hashes(artifacts_folder, expected_artifacts_folder, params_to_match)
    assert len(pairs) == artifacts_expected

    for run_hash, ref_run_hash in pairs:
        assert ref_run_hash is not None

        curr_artifacts_folder = artifacts_folder / run_hash
        curr_ref_artifacts_folder = expected_artifacts_folder / ref_run_hash

        for path in curr_artifacts_folder.glob(f"best_model_weights_*.pt"):
            fname = path.name
            pytest.helpers.verify_deeplearning_model(
                curr_artifacts_folder / fname, 
                curr_ref_artifacts_folder / fname
            )
        pytest.helpers.verify_reports(
            curr_artifacts_folder,
            curr_ref_artifacts_folder, 
            with_train=False,
            with_val=False,
            with_test=True,
        )
