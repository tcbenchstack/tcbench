import pytest
import pathlib

import tcbench
from tcbench.libtcdatasets import datasets_utils

DATASETS_ROOT_FOLDER = datasets_utils.get_datasets_root_folder()


@pytest.mark.parametrize(
    "dataset_name, min_pkts, split, expected",
    [
        (
            tcbench.DATASETS.UCDAVISICDM19,
            -1,
            None,
            pathlib.Path("ucdavis-icdm19/preprocessed/ucdavis-icdm19.parquet"),
        ),
        (
            tcbench.DATASETS.UCDAVISICDM19,
            -1,
            -1,
            pathlib.Path("ucdavis-icdm19/preprocessed/ucdavis-icdm19.parquet"),
        ),
        (
            tcbench.DATASETS.UCDAVISICDM19,
            -1,
            0,
            pathlib.Path("ucdavis-icdm19/preprocessed/imc23/train_split_0.parquet"),
        ),
        (
            tcbench.DATASETS.UCDAVISICDM19,
            -1,
            "train",
            pathlib.Path("ucdavis-icdm19/preprocessed/imc23/train_split_0.parquet"),
        ),
        (
            tcbench.DATASETS.UCDAVISICDM19,
            -1,
            "human",
            pathlib.Path("ucdavis-icdm19/preprocessed/imc23/test_split_human.parquet"),
        ),
        (
            tcbench.DATASETS.UCDAVISICDM19,
            -1,
            "script",
            pathlib.Path("ucdavis-icdm19/preprocessed/imc23/test_split_script.parquet"),
        ),
        ####
        (
            tcbench.DATASETS.UTMOBILENET21,
            -1,
            None,
            pathlib.Path("utmobilenet21/preprocessed/utmobilenet21.parquet"),
        ),
        (
            tcbench.DATASETS.UTMOBILENET21,
            10,
            None,
            pathlib.Path(
                "utmobilenet21/preprocessed/imc23/utmobilenet21_filtered_minpkts10.parquet"
            ),
        ),
        (
            tcbench.DATASETS.UTMOBILENET21,
            10,
            True,
            pathlib.Path(
                "utmobilenet21/preprocessed/imc23/utmobilenet21_filtered_minpkts10_splits.parquet"
            ),
        ),
        ####
        (
            tcbench.DATASETS.MIRAGE19,
            -1,
            None,
            pathlib.Path("mirage19/preprocessed/mirage19.parquet"),
        ),
        (
            tcbench.DATASETS.MIRAGE19,
            10,
            None,
            pathlib.Path(
                "mirage19/preprocessed/imc23/mirage19_filtered_minpkts10.parquet"
            ),
        ),
        (
            tcbench.DATASETS.MIRAGE19,
            10,
            True,
            pathlib.Path(
                "mirage19/preprocessed/imc23/mirage19_filtered_minpkts10_splits.parquet"
            ),
        ),
        ####
        (
            tcbench.DATASETS.MIRAGE22,
            -1,
            None,
            pathlib.Path("mirage22/preprocessed/mirage22.parquet"),
        ),
        (
            tcbench.DATASETS.MIRAGE22,
            10,
            None,
            pathlib.Path(
                "mirage22/preprocessed/imc23/mirage22_filtered_minpkts10.parquet"
            ),
        ),
        (
            tcbench.DATASETS.MIRAGE22,
            10,
            True,
            pathlib.Path(
                "mirage22/preprocessed/imc23/mirage22_filtered_minpkts10_splits.parquet"
            ),
        ),
        (
            tcbench.DATASETS.MIRAGE22,
            1000,
            None,
            pathlib.Path(
                "mirage22/preprocessed/imc23/mirage22_filtered_minpkts1000.parquet"
            ),
        ),
        (
            tcbench.DATASETS.MIRAGE22,
            1000,
            True,
            pathlib.Path(
                "mirage22/preprocessed/imc23/mirage22_filtered_minpkts1000_splits.parquet"
            ),
        ),
    ],
)
def test_get_dataset_parquet_filename(dataset_name, min_pkts, split, expected):
    path = datasets_utils.get_dataset_parquet_filename(dataset_name, min_pkts, split)

    expected = DATASETS_ROOT_FOLDER / expected
    assert path == expected
