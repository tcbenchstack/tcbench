import pandas as pd

import pytest

from tcbench import DATASETS
from tcbench.modeling.dataprep import (
    MODELING_DATASET_TYPE,
    MODELING_INPUT_REPR_TYPE,
    load_dataset,
    AugmentWhenLoadingDataset,
    MultiViewDataset,
)


@pytest.mark.parametrize(
    "dataset_name, kwargs, expected_samples_count_csv",
    [
        (
            DATASETS.UCDAVISICDM19,
            dict(
                split_idx=-1,
            ),
            pytest.DIR_RESOURCES / "ucdavis-icdm19_samples_count_preprocessed.csv",
        ),
        (
            DATASETS.UCDAVISICDM19,
            dict(
                split_idx=None,
            ),
            pytest.DIR_RESOURCES / "ucdavis-icdm19_samples_count_preprocessed.csv",
        ),
        (
            DATASETS.UCDAVISICDM19,
            dict(
                split_idx=0,
            ),
            pytest.DIR_RESOURCES / "ucdavis-icdm19_samples_count_train_split_0.csv",
        ),
        (
            DATASETS.UTMOBILENET21,
            dict(split_idx=0, dataset_minpkts=10),
            pytest.DIR_RESOURCES / "utmobilenet21_samples_count_minpkts10.csv",
        ),
        (
            DATASETS.MIRAGE19,
            dict(split_idx=0, dataset_minpkts=10),
            pytest.DIR_RESOURCES / "mirage19_samples_count_minpkts10.csv",
        ),
        (
            DATASETS.MIRAGE22,
            dict(split_idx=0, dataset_minpkts=10),
            pytest.DIR_RESOURCES / "mirage22_samples_count_minpkts10.csv",
        ),
        (
            DATASETS.MIRAGE22,
            dict(split_idx=0, dataset_minpkts=1000),
            pytest.DIR_RESOURCES / "mirage22_samples_count_minpkts1000.csv",
        ),
    ],
)
def test_load_dataset_train_val(dataset_name, kwargs, expected_samples_count_csv):
    dset_train, dset_val = load_dataset(
        dataset_name, MODELING_DATASET_TYPE.TRAIN_VAL, **kwargs
    )

    train_samples_count = dset_train.samples_count()
    val_samples_count = dset_val.samples_count()

    expected = pd.read_csv(expected_samples_count_csv).set_index("app")

    assert (train_samples_count.loc[expected.index] == expected["train"]).all()
    assert (val_samples_count.loc[expected.index] == expected["val"]).all()


@pytest.mark.parametrize(
    "dataset_name, kwargs, expected_samples_count_csv",
    [
        (
            DATASETS.UCDAVISICDM19,
            dict(
                split_idx=0,
                flow_representation=MODELING_INPUT_REPR_TYPE.PKTSERIES,
                max_n_pkts=10,
            ),
            pytest.DIR_RESOURCES / "ucdavis-icdm19_samples_count_train_split_0.csv",
        ),
        (
            DATASETS.UCDAVISICDM19,
            dict(
                split_idx=0,
                flow_representation=MODELING_INPUT_REPR_TYPE.PKTSERIES,
                max_n_pkts=30,
            ),
            pytest.DIR_RESOURCES / "ucdavis-icdm19_samples_count_train_split_0.csv",
        ),
    ],
)
def test_load_dataset_train_val_timeseries(
    dataset_name, kwargs, expected_samples_count_csv
):
    dset_train, dset_val = load_dataset(
        dataset_name, MODELING_DATASET_TYPE.TRAIN_VAL, **kwargs
    )

    # verify that 3 features (pkts size, iat and dir) are concatenated
    X, y = dset_train[0]
    assert X.shape[1] == 3 * kwargs["max_n_pkts"]
    X, y = dset_val[0]
    assert X.shape[1] == 3 * kwargs["max_n_pkts"]

    train_samples_count = dset_train.samples_count()
    val_samples_count = dset_val.samples_count()

    expected = pd.read_csv(expected_samples_count_csv).set_index("app")

    assert (train_samples_count.loc[expected.index] == expected["train"]).all()
    assert (val_samples_count.loc[expected.index] == expected["val"]).all()


@pytest.mark.parametrize(
    "dataset_name, kwargs, expected_samples_count_csv",
    [
        (
            DATASETS.UCDAVISICDM19,
            dict(
                split_idx=-1,
            ),
            pytest.DIR_RESOURCES / "ucdavis-icdm19_samples_count_test.csv",
        ),
        (
            DATASETS.UTMOBILENET21,
            dict(
                dataset_minpkts=10,
                split_idx=0,
            ),
            pytest.DIR_RESOURCES / "utmobilenet21_samples_count_minpkts10_test.csv",
        ),
        (
            DATASETS.MIRAGE19,
            dict(
                dataset_minpkts=10,
                split_idx=0,
            ),
            pytest.DIR_RESOURCES / "mirage19_samples_count_minpkts10_test.csv",
        ),
        (
            DATASETS.MIRAGE22,
            dict(
                dataset_minpkts=10,
                split_idx=0,
            ),
            pytest.DIR_RESOURCES / "mirage22_samples_count_minpkts10_test.csv",
        ),
        (
            DATASETS.MIRAGE22,
            dict(
                dataset_minpkts=1000,
                split_idx=0,
            ),
            pytest.DIR_RESOURCES / "mirage22_samples_count_minpkts1000_test.csv",
        ),
    ],
)
def test_load_dataset_test(dataset_name, kwargs, expected_samples_count_csv):
    dset_dict = load_dataset(dataset_name, MODELING_DATASET_TYPE.TEST, **kwargs)

    l = [
        pd.DataFrame(dset.samples_count()).rename({"count": name}, axis=1)
        for name, dset in dset_dict.items()
    ]
    if len(l) == 1:
        samples_count = l[0]
    else:
        samples_count = pd.concat(l, axis=1)

    expected = pd.read_csv(expected_samples_count_csv).set_index("app")

    assert (samples_count.loc[expected.index] == expected).all().all()


@pytest.mark.parametrize(
    "dataset_name, kwargs, expected_samples_count_csv",
    [
        (
            DATASETS.UCDAVISICDM19,
            dict(
                split_idx=0,
            ),
            pytest.DIR_RESOURCES / "ucdavis-icdm19_samples_count_leftover.csv",
        ),
    ],
)
def test_load_dataset_leftover(dataset_name, kwargs, expected_samples_count_csv):
    dset_train, dset_val = load_dataset(
        dataset_name, MODELING_DATASET_TYPE.TRAIN_VAL, **kwargs
    )

    kwargs_leftover = dict(
        dset_train=dset_train,
        dset_val=dset_val,
    )
    dset_leftover = load_dataset(
        dataset_name, MODELING_DATASET_TYPE.TRAIN_VAL_LEFTOVER, **kwargs_leftover
    )
    samples_count = dset_leftover.samples_count()
    expected = pd.read_csv(expected_samples_count_csv).set_index("app", drop=True)[
        "test"
    ]
    assert (samples_count.loc[expected.index] == expected).all()


@pytest.mark.parametrize(
    "dataset_name, kwargs, expected_samples_count_csv",
    [
        (
            DATASETS.UCDAVISICDM19,
            dict(
                train_samples=10,
            ),
            pytest.DIR_RESOURCES / "ucdavis-icdm19_samples_count_finetuning.csv",
        ),
    ],
)
def test_load_dataset_finetuning(dataset_name, kwargs, expected_samples_count_csv):
    dset_dict = load_dataset(dataset_name, MODELING_DATASET_TYPE.FINETUNING, **kwargs)

    expected = pd.read_csv(expected_samples_count_csv).set_index("app", drop=True)
    for dset_name, dset in dset_dict.items():
        samples_count = dset.samples_count()
        assert (samples_count.loc[expected.index] == expected[dset_name]).all()


@pytest.mark.parametrize(
    "dataset_name, kwargs",
    [
        (
            DATASETS.UCDAVISICDM19,
            dict(aug_config=None, aug_samples=10, aug_when_loading=True),
        ),
        (
            DATASETS.UCDAVISICDM19,
            dict(
                aug_config=dict(timeshift=None), aug_samples=10, aug_when_loading=True
            ),
        ),
        (
            DATASETS.UCDAVISICDM19,
            dict(
                aug_config=dict(changertt=None), aug_samples=10, aug_when_loading=True
            ),
        ),
        (
            DATASETS.UCDAVISICDM19,
            dict(
                aug_config=dict(packetloss=None), aug_samples=10, aug_when_loading=True
            ),
        ),
        (
            DATASETS.UCDAVISICDM19,
            dict(
                aug_config=dict(colorjitter=None), aug_samples=10, aug_when_loading=True
            ),
        ),
        (
            DATASETS.UCDAVISICDM19,
            dict(aug_config=dict(rotate=None), aug_samples=10, aug_when_loading=True),
        ),
        (
            DATASETS.UCDAVISICDM19,
            dict(
                aug_config=dict(horizontalflip=None),
                aug_samples=10,
                aug_when_loading=True,
            ),
        ),
        (
            DATASETS.UCDAVISICDM19,
            dict(aug_config=None, aug_samples=20, aug_when_loading=True),
        ),
        (
            DATASETS.UCDAVISICDM19,
            dict(
                aug_config=dict(timeshift=None), aug_samples=20, aug_when_loading=True
            ),
        ),
        (
            DATASETS.UCDAVISICDM19,
            dict(
                aug_config=dict(changertt=None), aug_samples=20, aug_when_loading=True
            ),
        ),
        (
            DATASETS.UCDAVISICDM19,
            dict(
                aug_config=dict(packetloss=None), aug_samples=20, aug_when_loading=True
            ),
        ),
        (
            DATASETS.UCDAVISICDM19,
            dict(
                aug_config=dict(colorjitter=None), aug_samples=20, aug_when_loading=True
            ),
        ),
        (
            DATASETS.UCDAVISICDM19,
            dict(aug_config=dict(rotate=None), aug_samples=20, aug_when_loading=True),
        ),
        (
            DATASETS.UCDAVISICDM19,
            dict(
                aug_config=dict(horizontalflip=None),
                aug_samples=20,
                aug_when_loading=True,
            ),
        ),
    ],
)
def test_load_dataset_augmentations_at_loading(dataset_name, kwargs):
    dset_train, dset_val = load_dataset(
        dataset_name, MODELING_DATASET_TYPE.TRAIN_VAL, **kwargs
    )

    assert isinstance(dset_train, AugmentWhenLoadingDataset)
    assert isinstance(dset_val, AugmentWhenLoadingDataset)

    # create reference dataset without augmentation
    kwargs2 = kwargs.copy()
    kwargs2["aug_config"] = None
    dset_train_noaug, dset_val_noaug = load_dataset(
        dataset_name, MODELING_DATASET_TYPE.TRAIN_VAL, **kwargs2
    )

    aug_name, aug_hparams = (
        ("noaug", None)
        if kwargs["aug_config"] is None
        else list(kwargs["aug_config"].items())[0]
    )
    aug_samples = kwargs["aug_samples"]
    if aug_name == "horizontalflip":
        aug_samples = 2

    if aug_name == "noaug":
        assert (dset_train.samples_count() == dset_train_noaug.samples_count()).all()
        assert (dset_val.samples_count() == dset_val_noaug.samples_count()).all()
    else:
        assert dset_train.df.shape[0] == (dset_train_noaug.df.shape[0] * aug_samples)
        assert dset_val.df.shape[0] == (dset_val_noaug.df.shape[0] * aug_samples)
        assert (dset_train.df["row_id"].value_counts() == aug_samples).all()
        assert (dset_val.df["row_id"].value_counts() == aug_samples).all()


@pytest.mark.parametrize(
    "dataset_name, kwargs",
    [
        (
            DATASETS.UCDAVISICDM19,
            dict(
                aug_config=dict(timeshift=None), aug_samples=2, aug_when_loading=False
            ),
        ),
        # the code enforces only 2 views
        (
            DATASETS.UCDAVISICDM19,
            dict(
                aug_config=dict(timeshift=None), aug_samples=10, aug_when_loading=False
            ),
        ),
        (
            DATASETS.UCDAVISICDM19,
            dict(
                aug_config=dict(timeshift=None),
                aug_samples=10,
                aug_when_loading=False,
                aug_yield_also_original=True,
            ),
        ),
        (
            DATASETS.UCDAVISICDM19,
            dict(
                aug_config=dict(timeshift=None),
                aug_samples=10,
                aug_when_loading=False,
                aug_yield_also_original=False,
            ),
        ),
    ],
)
def test_load_dataset_multiview(dataset_name, kwargs):
    dset_train, dset_val = load_dataset(
        dataset_name, MODELING_DATASET_TYPE.TRAIN_VAL, **kwargs
    )

    assert isinstance(dset_train, MultiViewDataset)
    assert isinstance(dset_val, MultiViewDataset)

    aug_yield_also_original = kwargs.get("aug_yield_also_original", False)
    X, y = dset_train[0]

    if not aug_yield_also_original:
        assert len(X) == 2
    else:
        assert len(X) == 3
