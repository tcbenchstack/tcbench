---
title: pytest
icon: simple/pytest
---

# ML unit testing

Multiple tests are available to verify different functionalities
for either tcbench and the modeling campaigns created.

Tests are not bundled with pypi installation. Rather, you need
to follow the procedure described in the [artifact page](/tcbench/papers/imc23/artifacts/)
to fetch the source code and install all artifacts and datasets.

Tests are coded via [`pytest` :simple-pytest:](https://docs.pytest.org/en/7.4.x/)
and are available under the `/tests` folder.

!!! warning "Tests trigger model training"

    Most of the test verify that the models train for
    the campaigns described in the paper are indeed reproducible, i.e.,
    the provide the exact same models obtained for the paper.

    To do so, the pytest resources fetched from figshare 
    contains a subset of reference models so the test
    trigger the modeling for those scenarios and check
    that what trained matches what created for the paper.

    So be aware that running these tests might take a while
    depending on your local environment.


To trigger all tests run

```
pytest tests
```

!!! note "Output"
    ```
    ============================ test session starts ======================================
    platform linux -- Python 3.10.13, pytest-7.4.2, pluggy-1.3.0
    rootdir: /tmp/tcbench-pip/tcbench
    plugins: anyio-3.7.1, helpers-namespace-2021.12.29
    collected 101 items

    tests/test_augmentations_at_loading.py ...........                               [ 10%]
    tests/test_augmentations_at_loading_xgboost.py .                                 [ 11%]
    tests/test_cli_command_campaign.py ....                                          [ 15%]
    tests/test_cli_command_singlerun.py ............                                 [ 27%]
    tests/test_contrastive_learning_and_finetune.py ..                               [ 29%]
    tests/test_libtcdatasets_datasets_utils.py .................                     [ 46%]
    tests/test_modeling_backbone.py ................                                 [ 62%]
    tests/test_modeling_dataprep.py ..................................               [ 96%]
    tests/test_modeling_methods.py ....                                              [100%]
    ============================== 101 passed, 8 warnings in 6523.55s (1:48:43) =========================
    ```
