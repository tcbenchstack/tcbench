Individual modeling run can be triggered
from the subcommand `run`

```
tcbench run --help
```

!!! info "Output"
    ```
     Usage: tcbench run [OPTIONS] COMMAND [ARGS]...

     Trigger an individual modeling run.

    ╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────╮
    │ --help      Show this message and exit.                                                               │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ Commands ────────────────────────────────────────────────────────────────────────────────────────────╮
    │ augment-at-loading        Modeling by applying data augmentation when loading the training set.       │
    │ contralearn-and-finetune  Modeling by pre-training via constrative learning and then finetune the     │
    │                           final classifier from the pre-trained model.                                │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ```

The submission focuses on two types of runs. As from the
help string of the sub-commands

* `augment-at-loading` applies data augmentation to 
boost the number of samples in the training set.

* `contralearn-and-finetune` pre-train a model (via [SimCLR](https://arxiv.org/abs/2002.05709))
and then uses it to finetune the final classifier.

Both run types are associated to a variety of parameters which, 
for readability, are organized in groups when printing the `--help`.

Each parameter help string should be sufficient for understanding their purpose
so we skip their detailed discussion.

Yet, for each run type we report a reference example of how to trigger it.

!!! tip "Runs and campaigns are [repeatable](https://www.acm.org/publications/policies/artifact-review-badging)"

    When executing the reference examples (or any of the runs
    collected in the ML artifacts) we expect you to obtain the 
    exact same results! :fontawesome-regular-face-smile:

    
## `augment-at-loading`

```
tcbench run augment-at-loading --help
```

!!! info "Output"
    ```
     Usage: tcbench run augment-at-loading [OPTIONS]

     Modeling by applying data augmentation when loading the training set.

    ╭─ General options ─────────────────────────────────────────────────────────────────────────────────────╮
    │ --artifacts-folder       PATH     Artifacts folder. [default: debug/artifacts]                        │
    │ --aim-repo               PATH     AIM repository location (local folder or URL). [default: debug]     │
    │ --aim-experiment-name    TEXT     The name of the experiment for AIM tracking.                        │
    │                                   [default: augmentation-at-loading]                                  │
    │ --gpu-index              TEXT     The id of the GPU to use (if training with deep learning).          │
    │                                   [default: 0]                                                        │
    │ --workers                INTEGER  Number of parallel worker for loading the data. [default: 20]       │
    │ --seed                   INTEGER  Seed to initialize random generators. [default: 12345]              │
    │ --help                            Show this message and exit.                                         │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ Data ────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ --dataset                   [ucdavis-icdm19|utmobilenet21|mirag  Dataset to use for modeling.         │
    │                             e19|mirage22]                        [default: ucdavis-icdm19]            │
    │ --dataset-minpkts           [-1|10|100|1000]                     In combination with --dataset,       │
    │                                                                  refines preprocessed and split       │
    │                                                                  dataset to use.                      │
    │                                                                  [default: -1]                        │
    │ --flowpic-dim               [32|64|1500]                         Flowpic dimension. [default: 32]     │
    │ --flowpic-block-duration    INTEGER                              Number of seconds for the head of a  │
    │                                                                  flow (i.e., block) to use for a      │
    │                                                                  flowpic.                             │
    │                                                                  [default: 15]                        │
    │ --split-index               INTEGER                              Data split index. [default: 0]       │
    │ --train-val-split-ratio     FLOAT                                If not predefined by the selected    │
    │                                                                  split, the ratio data to use for     │
    │                                                                  training (rest is for validation).   │
    │                                                                  [default: 0.8]                       │
    │ --aug-name                  [noaug|rotate|horizontalflip|colorj  Name of the augmentation to use.     │
    │                             itter|packetloss|timeshift|changert  [default: noaug]                     │
    │                             t]                                                                        │
    │ --no-test-leftover                                               Skip test on leftover split          │
    │                                                                  (specific for ucdavis-icdm19, and    │
    │                                                                  default enabled for all other        │
    │                                                                  datasets).                           │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ Modeling ────────────────────────────────────────────────────────────────────────────────────────────╮
    │ --method    [monolithic|xgboost]  Method to use for training. [default: monolithic]                   │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ DL hyper params ─────────────────────────────────────────────────────────────────────────────────────╮
    │ --batch-size        INTEGER  Training batch size [default: 32]                                        │
    │ --learning-rate     FLOAT    Training learning rate. [default: 0.001]                                 │
    │ --patience-steps    INTEGER  Max. number of epochs without improvement before stopping training.      │
    │                              [default: 5]                                                             │
    │ --epochs            INTEGER  Number of epochs for training. [default: 50]                             │
    │ --no-dropout                 Mask dropout layers with Identity layers.                                │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ XGBoost hyper params ────────────────────────────────────────────────────────────────────────────────╮
    │ --input-repr       TEXT     Input representation. [default: pktseries]                                │
    │ --pktseries-len    INTEGER  Number of packets (when using time series as input). [default: 10]        │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ```

### Reference example

```
tcbench run augment-at-loading \
    --dataset ucdavis-icdm19 \
    --learning-rate 0.001 \
    --batch-size 32 \
    --flowpic-dim 32 \
    --split-index 0 \
    --seed 12345 \
    --aug-name noaug \
    --method monolithic
```

??? info "Output"
    ```
    opened log at debug/artifacts/0179aa10fa7245d6bfd79b49/log.txt

    connecting to AIM repo at: debug
    created aim run hash=0179aa10fa7245d6bfd79b49
    artifacts folder at: debug/artifacts/0179aa10fa7245d6bfd79b49
    WARNING: the artifact folder is not a subfolder of the AIM repo
    --- run hparams ---
    flowpic_dim: 32
    flowpic_block_duration: 15
    split_index: 0
    max_samples_per_class: -1
    aug_name: noaug
    patience_steps: 5
    suppress_val_augmentation: False
    dataset: ucdavis-icdm19
    dataset_minpkts: -1
    seed: 12345
    with_dropout: True
    -------------------
    loaded: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/imc23/train_split_0.parquet
    no augmentation
    no augmentation
    dataset samples count
                   train  val
    app
    google-doc        80   20
    google-drive      80   20
    google-music      80   20
    google-search     80   20
    youtube           80   20
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1            [-1, 6, 28, 28]             156
                  ReLU-2            [-1, 6, 28, 28]               0
             MaxPool2d-3            [-1, 6, 14, 14]               0
                Conv2d-4           [-1, 16, 10, 10]           2,416
                  ReLU-5           [-1, 16, 10, 10]               0
             Dropout2d-6           [-1, 16, 10, 10]               0
             MaxPool2d-7             [-1, 16, 5, 5]               0
               Flatten-8                  [-1, 400]               0
                Linear-9                  [-1, 120]          48,120
                 ReLU-10                  [-1, 120]               0
               Linear-11                   [-1, 84]          10,164
                 ReLU-12                   [-1, 84]               0
            Dropout1d-13                   [-1, 84]               0
               Linear-14                    [-1, 5]             425
    ================================================================
    Total params: 61,281
    Trainable params: 61,281
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.13
    Params size (MB): 0.23
    Estimated Total Size (MB): 0.36
    ----------------------------------------------------------------
    ---
    WARNING: Detected Dropout layer!
    WARNING: During supervised training, the monitored train_acc will be inaccurate
    ---
    epoch:   0 | train_loss: 1.608132 | train_acc:  38.5% | val_loss: 1.609477 | val_acc:  95.0% | *
    epoch:   1 | train_loss: 1.337334 | train_acc:  49.2% | val_loss: 1.043962 | val_acc:  96.0% | *
    epoch:   2 | train_loss: 1.108509 | train_acc:  56.8% | val_loss: 0.708749 | val_acc:  95.0% | *
    epoch:   3 | train_loss: 1.100363 | train_acc:  54.0% | val_loss: 0.544368 | val_acc:  95.0% | *
    epoch:   4 | train_loss: 1.031712 | train_acc:  53.5% | val_loss: 0.389431 | val_acc:  97.0% | *
    epoch:   5 | train_loss: 0.987201 | train_acc:  57.5% | val_loss: 0.479318 | val_acc:  96.0%
    epoch:   6 | train_loss: 0.996780 | train_acc:  59.8% | val_loss: 0.341755 | val_acc:  97.0% | *
    epoch:   7 | train_loss: 1.044375 | train_acc:  53.8% | val_loss: 0.311472 | val_acc:  97.0% | *
    epoch:   8 | train_loss: 0.941040 | train_acc:  59.5% | val_loss: 0.364296 | val_acc:  98.0%
    epoch:   9 | train_loss: 0.951338 | train_acc:  58.8% | val_loss: 0.255973 | val_acc:  97.0% | *
    epoch:  10 | train_loss: 0.882188 | train_acc:  59.5% | val_loss: 0.257684 | val_acc:  98.0%
    epoch:  11 | train_loss: 0.946672 | train_acc:  54.8% | val_loss: 0.239257 | val_acc:  97.0% | *
    epoch:  12 | train_loss: 0.963688 | train_acc:  57.0% | val_loss: 0.215193 | val_acc:  97.0% | *
    epoch:  13 | train_loss: 0.934703 | train_acc:  56.8% | val_loss: 0.268320 | val_acc:  98.0%
    epoch:  14 | train_loss: 1.032247 | train_acc:  51.8% | val_loss: 0.192979 | val_acc:  97.0% | *
    epoch:  15 | train_loss: 0.968257 | train_acc:  56.8% | val_loss: 0.173423 | val_acc:  98.0% | *
    epoch:  16 | train_loss: 0.904505 | train_acc:  59.2% | val_loss: 0.234691 | val_acc:  96.0%
    epoch:  17 | train_loss: 0.957420 | train_acc:  57.8% | val_loss: 0.172733 | val_acc:  98.0%
    epoch:  18 | train_loss: 0.890696 | train_acc:  59.5% | val_loss: 0.179425 | val_acc:  97.0%
    epoch:  19 | train_loss: 0.933493 | train_acc:  58.5% | val_loss: 0.156731 | val_acc:  98.0% | *
    epoch:  20 | train_loss: 0.927721 | train_acc:  60.2% | val_loss: 0.243990 | val_acc:  96.0%
    epoch:  21 | train_loss: 1.007217 | train_acc:  55.0% | val_loss: 0.193065 | val_acc:  98.0%
    epoch:  22 | train_loss: 0.873772 | train_acc:  60.8% | val_loss: 0.149504 | val_acc:  98.0% | *
    epoch:  23 | train_loss: 0.967534 | train_acc:  56.8% | val_loss: 0.147503 | val_acc:  98.0% | *
    epoch:  24 | train_loss: 0.881966 | train_acc:  62.0% | val_loss: 0.167102 | val_acc:  98.0%
    epoch:  25 | train_loss: 0.871421 | train_acc:  59.2% | val_loss: 0.158940 | val_acc:  98.0%
    epoch:  26 | train_loss: 0.940876 | train_acc:  57.8% | val_loss: 0.135381 | val_acc:  98.0% | *
    epoch:  27 | train_loss: 0.865655 | train_acc:  61.8% | val_loss: 0.140646 | val_acc:  98.0%
    epoch:  28 | train_loss: 0.912683 | train_acc:  58.0% | val_loss: 0.115086 | val_acc:  98.0% | *
    epoch:  29 | train_loss: 0.915417 | train_acc:  57.2% | val_loss: 0.179121 | val_acc:  98.0%
    epoch:  30 | train_loss: 0.792390 | train_acc:  64.2% | val_loss: 0.105904 | val_acc:  97.0% | *
    epoch:  31 | train_loss: 0.857908 | train_acc:  62.8% | val_loss: 0.120300 | val_acc:  98.0%
    epoch:  32 | train_loss: 0.884288 | train_acc:  62.2% | val_loss: 0.113660 | val_acc:  98.0%
    epoch:  33 | train_loss: 0.938388 | train_acc:  56.0% | val_loss: 0.107200 | val_acc:  98.0%
    epoch:  34 | train_loss: 0.845021 | train_acc:  62.2% | val_loss: 0.145654 | val_acc:  98.0%
    epoch:  35 | train_loss: 0.937906 | train_acc:  56.5% | val_loss: 0.102596 | val_acc:  98.0% | *
    epoch:  36 | train_loss: 0.920095 | train_acc:  58.8% | val_loss: 0.105251 | val_acc:  98.0%
    epoch:  37 | train_loss: 0.902691 | train_acc:  59.0% | val_loss: 0.136474 | val_acc:  98.0%
    epoch:  38 | train_loss: 0.883385 | train_acc:  59.8% | val_loss: 0.117455 | val_acc:  98.0%
    epoch:  39 | train_loss: 0.854439 | train_acc:  61.0% | val_loss: 0.103810 | val_acc:  97.0%
    epoch:  40 | train_loss: 0.854355 | train_acc:  61.5% | val_loss: 0.120359 | val_acc:  98.0%
    run out of patience

    ---train reports---

                   precision  recall  f1-score  support
    google-doc      0.975610   1.000  0.987654   80.000
    google-drive    1.000000   0.900  0.947368   80.000
    google-music    0.917647   0.975  0.945455   80.000
    google-search   0.987654   1.000  0.993789   80.000
    youtube         1.000000   1.000  1.000000   80.000
    accuracy        0.975000   0.975  0.975000    0.975
    macro avg       0.976182   0.975  0.974853  400.000
    weighted avg    0.976182   0.975  0.974853  400.000

                   google-doc  google-drive  google-music  google-search  youtube
    google-doc             80             0             0              0        0
    google-drive            1            72             7              0        0
    google-music            1             0            78              1        0
    google-search           0             0             0             80        0
    youtube                 0             0             0              0       80

    saving: debug/artifacts/0179aa10fa7245d6bfd79b49/train_class_rep.csv
    saving: debug/artifacts/0179aa10fa7245d6bfd79b49/train_conf_mtx.csv

    ---val reports---

                   precision  recall  f1-score  support
    google-doc      0.952381    1.00  0.975610    20.00
    google-drive    1.000000    0.95  0.974359    20.00
    google-music    0.952381    1.00  0.975610    20.00
    google-search   1.000000    0.95  0.974359    20.00
    youtube         1.000000    1.00  1.000000    20.00
    accuracy        0.980000    0.98  0.980000     0.98
    macro avg       0.980952    0.98  0.979987   100.00
    weighted avg    0.980952    0.98  0.979987   100.00

                   google-doc  google-drive  google-music  google-search  youtube
    google-doc             20             0             0              0        0
    google-drive            0            19             1              0        0
    google-music            0             0            20              0        0
    google-search           1             0             0             19        0
    youtube                 0             0             0              0       20

    saving: debug/artifacts/0179aa10fa7245d6bfd79b49/val_class_rep.csv
    saving: debug/artifacts/0179aa10fa7245d6bfd79b49/val_conf_mtx.csv
    loading: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/imc23/test_split_human.parquet
    loading: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/imc23/test_split_script.parquet
                   human  script
    app
    youtube           20      30
    google-drive      18      30
    google-doc        15      30
    google-music      15      30
    google-search     15      30
    Test dataset human | loss: 1.471955 | acc: 68.7

    ---test-human reports---

                   precision    recall  f1-score    support
    google-doc      0.500000  1.000000  0.666667  15.000000
    google-drive    0.736842  0.777778  0.756757  18.000000
    google-music    0.764706  0.866667  0.812500  15.000000
    google-search   0.000000  0.000000  0.000000  15.000000
    youtube         0.937500  0.750000  0.833333  20.000000
    accuracy        0.686747  0.686747  0.686747   0.686747
    macro avg       0.587810  0.678889  0.613851  83.000000
    weighted avg    0.614262  0.686747  0.632238  83.000000

                   google-doc  google-drive  google-music  google-search  youtube
    google-doc             15             0             0              0        0
    google-drive            0            14             3              0        1
    google-music            0             1            13              1        0
    google-search          15             0             0              0        0
    youtube                 0             4             1              0       15

    saving: debug/artifacts/0179aa10fa7245d6bfd79b49/test-human_class_rep.csv
    saving: debug/artifacts/0179aa10fa7245d6bfd79b49/test-human_conf_mtx.csv
    Test dataset script | loss: 0.178018 | acc: 95.3

    ---test-script reports---

                   precision    recall  f1-score     support
    google-doc      0.882353  1.000000  0.937500   30.000000
    google-drive    1.000000  0.900000  0.947368   30.000000
    google-music    0.933333  0.933333  0.933333   30.000000
    google-search   1.000000  0.933333  0.965517   30.000000
    youtube         0.967742  1.000000  0.983607   30.000000
    accuracy        0.953333  0.953333  0.953333    0.953333
    macro avg       0.956686  0.953333  0.953465  150.000000
    weighted avg    0.956686  0.953333  0.953465  150.000000

                   google-doc  google-drive  google-music  google-search  youtube
    google-doc             30             0             0              0        0
    google-drive            1            27             2              0        0
    google-music            1             0            28              0        1
    google-search           2             0             0             28        0
    youtube                 0             0             0              0       30

    saving: debug/artifacts/0179aa10fa7245d6bfd79b49/test-script_class_rep.csv
    saving: debug/artifacts/0179aa10fa7245d6bfd79b49/test-script_conf_mtx.csv
    loaded: /opt/anaconda/anaconda3/envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/ucdavis-icdm19.parquet

    Test dataset train-val-leftover | loss: 0.155867 | acc: 96.3


    ---test-train-val-leftover reports---

                   precision    recall  f1-score     support
    google-doc      0.953997  0.999210  0.976080  1266.00000
    google-drive    0.994190  0.915577  0.953265  1682.00000
    google-music    0.857143  0.960754  0.905996   637.00000
    google-search   0.979114  0.980612  0.979862  1960.00000
    youtube         0.969643  0.963620  0.966622  1127.00000
    accuracy        0.962980  0.962980  0.962980     0.96298
    macro avg       0.950817  0.963955  0.956365  6672.00000
    weighted avg    0.964904  0.962980  0.963151  6672.00000

                   google-doc  google-drive  google-music  google-search  youtube
    google-doc           1265             0             1              0        0
    google-drive           25          1540            98              0       19
    google-music            3             4           612              8       10
    google-search          32             1             0           1922        5
    youtube                 1             4             3             33     1086

    saving: debug/artifacts/0179aa10fa7245d6bfd79b49/test-train-val-leftover_class_rep.csv
    saving: debug/artifacts/0179aa10fa7245d6bfd79b49/test-train-val-leftover_conf_mtx.csv
    saving: debug/artifacts/0179aa10fa7245d6bfd79b49/params.yml
    ```


## `contralearn-and-finetune`

```
tcbench run contralearn-and-finetune --help
```

!!! info "Output"
    ```
     Usage: tcbench run contralearn-and-finetune [OPTIONS]

     Modeling by pre-training via constrative learning and then finetune the final
     classifier from the pre-trained model.

    ╭─ General options ─────────────────────────────────────────────────────────────────────────────────────╮
    │ --artifacts-folder       PATH     Artifacts folder. [default: debug/artifacts]                        │
    │ --aim-repo               PATH     AIM repository location (local folder or URL). [default: debug]     │
    │ --aim-experiment-name    TEXT     The name of the experiment for AIM tracking.                        │
    │                                   [default: contrastive-learning-and-finetune]                        │
    │ --gpu-index              TEXT     The id of the GPU to use (if training with deep learning).          │
    │                                   [default: 0]                                                        │
    │ --workers                INTEGER  Number of parallel worker for loading the data. [default: 20]       │
    │ --help                            Show this message and exit.                                         │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ Data ────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ --dataset                   [ucdavis-icdm19]  Dataset to use for modeling. [default: ucdavis-icdm19]  │
    │ --flowpic-dim               [32]              Flowpic dimension. [default: 32]                        │
    │ --flowpic-block-duration    INTEGER           Number of seconds for the head of a flow (i.e., block)  │
    │                                               to use for a flowpic.                                   │
    │                                               [default: 15]                                           │
    │ --split-index               INTEGER           Data split index. [default: 0]                          │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ General Deeplearning hyperparams ────────────────────────────────────────────────────────────────────╮
    │ --batch-size    INTEGER  Training batch size [default: 32]                                            │
    │ --no-dropout             Mask dropout layers with Identity layers.                                    │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ Contrastive learning hyperparams ────────────────────────────────────────────────────────────────────╮
    │ --cl-aug-names               TEXT     Coma separated list of augmentations pool for contrastive       │
    │                                       learning.                                                       │
    │                                       [default: changertt,timeshift]                                  │
    │ --cl-projection-layer-dim    INTEGER  The number of units in the contrastive learning projection      │
    │                                       layer.                                                          │
    │                                       [default: 30]                                                   │
    │ --cl-learning-rate           FLOAT    Learning rate for pretraining. [default: 0.001]                 │
    │ --cl-seed                    INTEGER  Seed for contrastive learning pretraining. [default: 12345]     │
    │ --cl-patience-steps          INTEGER  Max steps to wait before stopping training if the top5          │
    │                                       validation accuracy does not improve.                           │
    │                                       [default: 3]                                                    │
    │ --cl-temperature             FLOAT    Temperature for InfoNCE loss. [default: 0.07]                   │
    │ --cl-epochs                  INTEGER  Epochs for contrastive learning pretraining. [default: 50]      │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ Finetune hyperparams ────────────────────────────────────────────────────────────────────────────────╮
    │ --ft-learning-rate         FLOAT    Learning rate for finetune. [default: 0.01]                       │
    │ --ft-patience-steps        INTEGER  Max steps to wait before stopping finetune training loss does not │
    │                                     improve.                                                          │
    │                                     [default: 5]                                                      │
    │ --ft-patience-min-delta    FLOAT    Minimum decrease of training loss to be considered as             │
    │                                     improvement.                                                      │
    │                                     [default: 0.001]                                                  │
    │ --ft-train-samples         INTEGER  Number of samples per-class for finetune training. [default: 10]  │
    │ --ft-epochs                INTEGER  Epochs for finetune training. [default: 50]                       │
    │ --ft-seed                  INTEGER  Seed for finetune training. [default: 12345]                      │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ```

### Reference example
```
tcbench run contralearn-and-finetune 
    --dataset ucdavis-icdm19 \
    --batch-size 32 \
    --flowpic-dim 32 \
    --split-index 0 \
    --no-dropout \
    --cl-seed 12345 \
    --ft-seed 12345 \
    --cl-projection-layer-dim 30
```

??? info "Output"
    ```
    opened log at debug/artifacts/ucdavis-icdm19/b6255a30a35d4f5daa0beab1/log.txt

    connecting to AIM repo at: debug
    created aim run hash=b6255a30a35d4f5daa0beab1
    artifacts folder at: debug/artifacts/ucdavis-icdm19/b6255a30a35d4f5daa0beab1
    WARNING: the artifact folder is not a subfolder of the AIM repo
    --- run hparams ---
    flowpic_dim: 32
    split_index: 0
    dataset: ucdavis-icdm19
    dataset_minpkts: -1
    contrastive_learning_seed: 12345
    finetune_seed: 12345
    finetune_train_samples: 10
    with_dropout: False
    projection_layer_dim: 30
    finetune_augmentation: none
    augmentations: ['changertt', 'timeshift']
    -------------------
    loaded: /opt/anaconda/anaconda3/envs/tcbench-johndoe/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/imc23/train_split_0.parquet
    dataset samples count
                   train  val
    app
    google-doc        80   20
    google-drive      80   20
    google-music      80   20
    google-search     80   20
    youtube           80   20

    ==== network adapted for pretrain ====
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1            [-1, 6, 28, 28]             156
                  ReLU-2            [-1, 6, 28, 28]               0
             MaxPool2d-3            [-1, 6, 14, 14]               0
                Conv2d-4           [-1, 16, 10, 10]           2,416
                  ReLU-5           [-1, 16, 10, 10]               0
              Identity-6           [-1, 16, 10, 10]               0
             MaxPool2d-7             [-1, 16, 5, 5]               0
               Flatten-8                  [-1, 400]               0
                Linear-9                  [-1, 120]          48,120
                 ReLU-10                  [-1, 120]               0
               Linear-11                  [-1, 120]          14,520
                 ReLU-12                  [-1, 120]               0
             Identity-13                  [-1, 120]               0
               Linear-14                   [-1, 30]           3,630
    ================================================================
    Total params: 68,842
    Trainable params: 68,842
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.13
    Params size (MB): 0.26
    Estimated Total Size (MB): 0.39
    ----------------------------------------------------------------
    epoch:   0 | train_loss: 2.792354 | train_acc_top_1:  22.1% | train_acc_top_5:  58.2% | val_loss: 2.770429 | val_acc_top_1:  24.6% | val_acc_top_5:  51.2% | *
    epoch:   1 | train_loss: 2.389633 | train_acc_top_1:  27.4% | train_acc_top_5:  62.3% | val_loss: 2.722881 | val_acc_top_1:  31.2% | val_acc_top_5:  54.3% | *
    epoch:   2 | train_loss: 2.277134 | train_acc_top_1:  26.1% | train_acc_top_5:  64.1% | val_loss: 2.687306 | val_acc_top_1:  26.6% | val_acc_top_5:  52.3%
    epoch:   3 | train_loss: 2.155350 | train_acc_top_1:  30.5% | train_acc_top_5:  68.5% | val_loss: 2.353693 | val_acc_top_1:  25.4% | val_acc_top_5:  60.5% | *
    epoch:   4 | train_loss: 2.128647 | train_acc_top_1:  30.8% | train_acc_top_5:  70.3% | val_loss: 2.409730 | val_acc_top_1:  30.9% | val_acc_top_5:  57.8%
    epoch:   5 | train_loss: 2.033960 | train_acc_top_1:  32.5% | train_acc_top_5:  72.5% | val_loss: 2.468154 | val_acc_top_1:  30.5% | val_acc_top_5:  57.8%
    epoch:   6 | train_loss: 1.933769 | train_acc_top_1:  36.7% | train_acc_top_5:  74.2% | val_loss: 2.246096 | val_acc_top_1:  34.8% | val_acc_top_5:  64.1% | *
    epoch:   7 | train_loss: 1.913906 | train_acc_top_1:  37.7% | train_acc_top_5:  78.6% | val_loss: 2.082685 | val_acc_top_1:  45.7% | val_acc_top_5:  68.4% | *
    epoch:   8 | train_loss: 1.800368 | train_acc_top_1:  36.2% | train_acc_top_5:  81.1% | val_loss: 2.157686 | val_acc_top_1:  35.2% | val_acc_top_5:  67.6%
    epoch:   9 | train_loss: 1.739876 | train_acc_top_1:  43.5% | train_acc_top_5:  81.7% | val_loss: 2.286904 | val_acc_top_1:  39.5% | val_acc_top_5:  67.6%
    epoch:  10 | train_loss: 1.773573 | train_acc_top_1:  41.8% | train_acc_top_5:  79.8% | val_loss: 2.208454 | val_acc_top_1:  30.5% | val_acc_top_5:  68.4%
    run out of patience
    saving: debug/artifacts/ucdavis-icdm19/b6255a30a35d4f5daa0beab1/best_model_weights_pretrain_split_0.pt
                   human_test  human_train  script_test  script_train
    app
    google-doc              5           10           20            10
    google-drive            8           10           20            10
    google-music            5           10           20            10
    google-search           5           10           20            10
    youtube                10           10           20            10

    --- finetune (train) on human ---
    app
    google-doc       10
    google-drive     10
    google-music     10
    google-search    10
    youtube          10
    Name: count, dtype: int64

    ==== network adapted for fine-tuning ====
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1            [-1, 6, 28, 28]             156
                  ReLU-2            [-1, 6, 28, 28]               0
             MaxPool2d-3            [-1, 6, 14, 14]               0
                Conv2d-4           [-1, 16, 10, 10]           2,416
                  ReLU-5           [-1, 16, 10, 10]               0
              Identity-6           [-1, 16, 10, 10]               0
             MaxPool2d-7             [-1, 16, 5, 5]               0
               Flatten-8                  [-1, 400]               0
                Linear-9                  [-1, 120]          48,120
                 ReLU-10                  [-1, 120]               0
             Identity-11                  [-1, 120]               0
             Identity-12                  [-1, 120]               0
             Identity-13                  [-1, 120]               0
               Linear-14                    [-1, 5]             605
    ================================================================
    Total params: 51,297
    Trainable params: 51,297
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.13
    Params size (MB): 0.20
    Estimated Total Size (MB): 0.33
    ----------------------------------------------------------------
    epoch:   0 | train_loss: 3.231672 | train_acc:   8.0% | *
    epoch:   1 | train_loss: 3.074481 | train_acc:  32.0% | *
    epoch:   2 | train_loss: 2.968672 | train_acc:  40.0% | *
    epoch:   3 | train_loss: 2.879148 | train_acc:  40.0% | *
    epoch:   4 | train_loss: 2.783693 | train_acc:  42.0% | *
    epoch:   5 | train_loss: 2.705268 | train_acc:  36.0% | *
    epoch:   6 | train_loss: 2.661375 | train_acc:  36.0% | *
    epoch:   7 | train_loss: 2.562433 | train_acc:  38.0% | *
    epoch:   8 | train_loss: 2.525108 | train_acc:  40.0% | *
    epoch:   9 | train_loss: 2.476060 | train_acc:  42.0% | *
    epoch:  10 | train_loss: 2.450299 | train_acc:  46.0% | *
    epoch:  11 | train_loss: 2.396669 | train_acc:  50.0% | *
    epoch:  12 | train_loss: 2.376229 | train_acc:  58.0% | *
    epoch:  13 | train_loss: 2.351827 | train_acc:  64.0% | *
    epoch:  14 | train_loss: 2.383254 | train_acc:  64.0%
    epoch:  15 | train_loss: 2.248392 | train_acc:  64.0% | *
    epoch:  16 | train_loss: 2.223375 | train_acc:  64.0% | *
    epoch:  17 | train_loss: 2.243016 | train_acc:  66.0%
    epoch:  18 | train_loss: 2.158619 | train_acc:  68.0% | *
    epoch:  19 | train_loss: 2.122373 | train_acc:  70.0% | *
    epoch:  20 | train_loss: 2.078161 | train_acc:  72.0% | *
    epoch:  21 | train_loss: 2.087560 | train_acc:  72.0%
    epoch:  22 | train_loss: 2.053180 | train_acc:  72.0% | *
    epoch:  23 | train_loss: 1.953339 | train_acc:  76.0% | *
    epoch:  24 | train_loss: 2.032256 | train_acc:  80.0%
    epoch:  25 | train_loss: 1.967944 | train_acc:  82.0%
    epoch:  26 | train_loss: 2.018130 | train_acc:  82.0%
    epoch:  27 | train_loss: 1.859777 | train_acc:  82.0% | *
    epoch:  28 | train_loss: 1.923519 | train_acc:  82.0%
    epoch:  29 | train_loss: 1.845623 | train_acc:  82.0% | *
    epoch:  30 | train_loss: 1.927216 | train_acc:  82.0%
    epoch:  31 | train_loss: 1.904452 | train_acc:  82.0%
    epoch:  32 | train_loss: 1.835915 | train_acc:  84.0% | *
    epoch:  33 | train_loss: 1.840065 | train_acc:  84.0%
    epoch:  34 | train_loss: 1.829454 | train_acc:  84.0% | *
    epoch:  35 | train_loss: 1.747133 | train_acc:  84.0% | *
    epoch:  36 | train_loss: 1.829719 | train_acc:  84.0%
    epoch:  37 | train_loss: 1.773501 | train_acc:  84.0%
    epoch:  38 | train_loss: 1.727183 | train_acc:  84.0% | *
    epoch:  39 | train_loss: 1.750620 | train_acc:  84.0%
    epoch:  40 | train_loss: 1.795435 | train_acc:  82.0%
    epoch:  41 | train_loss: 1.750014 | train_acc:  82.0%
    epoch:  42 | train_loss: 1.681466 | train_acc:  84.0% | *
    epoch:  43 | train_loss: 1.620332 | train_acc:  84.0% | *
    epoch:  44 | train_loss: 1.638735 | train_acc:  84.0%
    epoch:  45 | train_loss: 1.647057 | train_acc:  84.0%
    epoch:  46 | train_loss: 1.627543 | train_acc:  84.0%
    epoch:  47 | train_loss: 1.693797 | train_acc:  84.0%
    epoch:  48 | train_loss: 1.609901 | train_acc:  86.0% | *
    epoch:  49 | train_loss: 1.628688 | train_acc:  86.0%
    reached max epochs
    saving: debug/artifacts/ucdavis-icdm19/b6255a30a35d4f5daa0beab1/best_model_weights_finetune_human_from_split_0.pt

    --- finetune (test) on human ---
    app
    youtube          10
    google-drive      8
    google-doc        5
    google-music      5
    google-search     5
    Name: count, dtype: int64
    Test dataset human | loss: 2.244415 | acc: 75.8

    ---test-human reports---

                   precision    recall  f1-score    support
    google-doc      0.714286  1.000000  0.833333   5.000000
    google-drive    0.600000  0.750000  0.666667   8.000000
    google-music    0.714286  1.000000  0.833333   5.000000
    google-search   1.000000  0.600000  0.750000   5.000000
    youtube         1.000000  0.600000  0.750000  10.000000
    accuracy        0.757576  0.757576  0.757576   0.757576
    macro avg       0.805714  0.790000  0.766667  33.000000
    weighted avg    0.816450  0.757576  0.755051  33.000000

                   google-doc  google-drive  google-music  google-search  youtube
    google-doc              5             0             0              0        0
    google-drive            0             6             2              0        0
    google-music            0             0             5              0        0
    google-search           2             0             0              3        0
    youtube                 0             4             0              0        6

    saving: debug/artifacts/ucdavis-icdm19/b6255a30a35d4f5daa0beab1/test-human_class_rep.csv
    saving: debug/artifacts/ucdavis-icdm19/b6255a30a35d4f5daa0beab1/test-human_conf_mtx.csv

    --- finetune (train) on script ---
    app
    google-doc       10
    google-drive     10
    google-music     10
    google-search    10
    youtube          10
    Name: count, dtype: int64

    ==== network adapted for fine-tuning ====
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1            [-1, 6, 28, 28]             156
                  ReLU-2            [-1, 6, 28, 28]               0
             MaxPool2d-3            [-1, 6, 14, 14]               0
                Conv2d-4           [-1, 16, 10, 10]           2,416
                  ReLU-5           [-1, 16, 10, 10]               0
              Identity-6           [-1, 16, 10, 10]               0
             MaxPool2d-7             [-1, 16, 5, 5]               0
               Flatten-8                  [-1, 400]               0
                Linear-9                  [-1, 120]          48,120
                 ReLU-10                  [-1, 120]               0
             Identity-11                  [-1, 120]               0
             Identity-12                  [-1, 120]               0
             Identity-13                  [-1, 120]               0
               Linear-14                    [-1, 5]             605
    ================================================================
    Total params: 51,297
    Trainable params: 51,297
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.13
    Params size (MB): 0.20
    Estimated Total Size (MB): 0.33
    ----------------------------------------------------------------
    epoch:   0 | train_loss: 3.220116 | train_acc:  20.0% | *
    epoch:   1 | train_loss: 3.102323 | train_acc:  34.0% | *
    epoch:   2 | train_loss: 2.989245 | train_acc:  44.0% | *
    epoch:   3 | train_loss: 2.864427 | train_acc:  46.0% | *
    epoch:   4 | train_loss: 2.747027 | train_acc:  52.0% | *
    epoch:   5 | train_loss: 2.699277 | train_acc:  54.0% | *
    epoch:   6 | train_loss: 2.589165 | train_acc:  56.0% | *
    epoch:   7 | train_loss: 2.562590 | train_acc:  70.0% | *
    epoch:   8 | train_loss: 2.483140 | train_acc:  72.0% | *
    epoch:   9 | train_loss: 2.362642 | train_acc:  76.0% | *
    epoch:  10 | train_loss: 2.249433 | train_acc:  78.0% | *
    epoch:  11 | train_loss: 2.289871 | train_acc:  78.0%
    epoch:  12 | train_loss: 2.206161 | train_acc:  78.0% | *
    epoch:  13 | train_loss: 2.123558 | train_acc:  78.0% | *
    epoch:  14 | train_loss: 2.060765 | train_acc:  76.0% | *
    epoch:  15 | train_loss: 2.060799 | train_acc:  76.0%
    epoch:  16 | train_loss: 1.909777 | train_acc:  76.0% | *
    epoch:  17 | train_loss: 1.910333 | train_acc:  76.0%
    epoch:  18 | train_loss: 1.919481 | train_acc:  80.0%
    epoch:  19 | train_loss: 1.793242 | train_acc:  84.0% | *
    epoch:  20 | train_loss: 1.772380 | train_acc:  92.0% | *
    epoch:  21 | train_loss: 1.818070 | train_acc:  92.0%
    epoch:  22 | train_loss: 1.777959 | train_acc:  92.0%
    epoch:  23 | train_loss: 1.720130 | train_acc:  92.0% | *
    epoch:  24 | train_loss: 1.641118 | train_acc:  92.0% | *
    epoch:  25 | train_loss: 1.600633 | train_acc:  92.0% | *
    epoch:  26 | train_loss: 1.522475 | train_acc:  92.0% | *
    epoch:  27 | train_loss: 1.518417 | train_acc:  92.0% | *
    epoch:  28 | train_loss: 1.578722 | train_acc:  92.0%
    epoch:  29 | train_loss: 1.502436 | train_acc:  92.0% | *
    epoch:  30 | train_loss: 1.487650 | train_acc:  94.0% | *
    epoch:  31 | train_loss: 1.516318 | train_acc:  94.0%
    epoch:  32 | train_loss: 1.452326 | train_acc:  92.0% | *
    epoch:  33 | train_loss: 1.400150 | train_acc:  92.0% | *
    epoch:  34 | train_loss: 1.314930 | train_acc:  92.0% | *
    epoch:  35 | train_loss: 1.313898 | train_acc:  92.0% | *
    epoch:  36 | train_loss: 1.343631 | train_acc:  92.0%
    epoch:  37 | train_loss: 1.351552 | train_acc:  92.0%
    epoch:  38 | train_loss: 1.242814 | train_acc:  92.0% | *
    epoch:  39 | train_loss: 1.285116 | train_acc:  92.0%
    epoch:  40 | train_loss: 1.248655 | train_acc:  92.0%
    epoch:  41 | train_loss: 1.279359 | train_acc:  92.0%
    epoch:  42 | train_loss: 1.189178 | train_acc:  92.0% | *
    epoch:  43 | train_loss: 1.235253 | train_acc:  92.0%
    epoch:  44 | train_loss: 1.195394 | train_acc:  92.0%
    epoch:  45 | train_loss: 1.197707 | train_acc:  92.0%
    epoch:  46 | train_loss: 1.259920 | train_acc:  92.0%
    epoch:  47 | train_loss: 1.193645 | train_acc:  92.0%
    run out of patience
    saving: debug/artifacts/ucdavis-icdm19/b6255a30a35d4f5daa0beab1/best_model_weights_finetune_script_from_split_0.pt

    --- finetune (test) on script ---
    app
    google-doc       20
    google-drive     20
    google-music     20
    google-search    20
    youtube          20
    Name: count, dtype: int64
    Test dataset script | loss: 0.754386 | acc: 93.0

    ---test-script reports---

                   precision  recall  f1-score  support
    google-doc      0.900000    0.90  0.900000    20.00
    google-drive    1.000000    0.90  0.947368    20.00
    google-music    0.904762    0.95  0.926829    20.00
    google-search   0.904762    0.95  0.926829    20.00
    youtube         0.950000    0.95  0.950000    20.00
    accuracy        0.930000    0.93  0.930000     0.93
    macro avg       0.931905    0.93  0.930205   100.00
    weighted avg    0.931905    0.93  0.930205   100.00

                   google-doc  google-drive  google-music  google-search  youtube
    google-doc             18             0             0              2        0
    google-drive            0            18             1              0        1
    google-music            1             0            19              0        0
    google-search           1             0             0             19        0
    youtube                 0             0             1              0       19

    saving: debug/artifacts/ucdavis-icdm19/b6255a30a35d4f5daa0beab1/test-script_class_rep.csv
    saving: debug/artifacts/ucdavis-icdm19/b6255a30a35d4f5daa0beab1/test-script_conf_mtx.csv
    saving: debug/artifacts/ucdavis-icdm19/b6255a30a35d4f5daa0beab1/params.yml
    ```
