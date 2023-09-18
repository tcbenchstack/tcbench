import pandas as pd

import pytest
import pathlib

import tcbench
from tcbench.modeling import (
    utils,
    run_augmentations_at_loading_xgboost,
    MODELING_DATASET_TYPE,
)


@pytest.mark.parametrize(
    "params, expected_artifacts_folder",
    [
        (
            [
                f"--dataset {str(tcbench.DATASETS.UCDAVISICDM19)}",
                "--flow-representation pktseries",
                "--max-n-pkts 10",
                "--split-index 0",
                "--seed 12345",
            ],
            pytest.DIR_RESOURCES
            / pathlib.Path(
                "_reference_aim_run/ucdavis-icdm19/xgboost/noaugmentation-timeseries/5fa59c129a3e4aa6bb9b7640"
            ),
        ),
    ],
)
def test_main(tmp_path, params, expected_artifacts_folder):
    params.append(f"--artifacts-folder {tmp_path}/artifacts")
    params.append(f"--aim-repo {tmp_path}")

    parser = run_augmentations_at_loading_xgboost.cli_parser()
    args = parser.parse_args((" ".join(params)).split())

    state = run_augmentations_at_loading_xgboost.main(args)

    # the output folder is based on the aim run hash
    artifacts_folder = next((tmp_path / 'artifacts').iterdir())

    # verifying model files
    fname = f"xgb_model_split_{args.split_index}.json"
#    pytest.helpers.verify_md5_model(
#        artifacts_folder / fname, expected_artifacts_folder / fname
#    )

    pytest.helpers.verify_reports(artifacts_folder, expected_artifacts_folder)
