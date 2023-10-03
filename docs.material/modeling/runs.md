---
title: Runs
icon: material/cube
---

Individual modeling run can be triggered
using the subcommand `run`

```
tcbench run --help
```

!!! info "Output"
    ```
	 Usage: tcbench run [OPTIONS] COMMAND [ARGS]...

	 Triggers a modeling run.

	╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
	│ --help      Show this message and exit.                                                          │
	╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
	╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────╮
	│ augment-at-loading        Modeling by applying data augmentation when loading the training set.  │
	│ contralearn-and-finetune  Modeling by pre-training via constrative learning and then finetune    │
	│                           the final classifier from the pre-trained model.                       │
	╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
    ```


As from the help string of the sub-commands

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

	╭─ General options ────────────────────────────────────────────────────────────────────────────────╮
	│ --aim-experiment-name    TEXT     The name of the experiment for AIM tracking.                   │
	│                                   [default: augmentation-at-loading]                             │
	│ --aim-repo               PATH     AIM repository location (local folder or URL).                 │
	│                                   [default: aim-repo]                                            │
	│ --artifacts-folder       PATH     Artifacts folder. [default: aim-repo/artifacts]                │
	│ --gpu-index              TEXT     The id of the GPU to use (if training with deep learning).     │
	│                                   [default: 0]                                                   │
	│ --workers                INTEGER  Number of parallel worker for loading the data. [default: 20]  │
	│ --seed                   INTEGER  Seed to initialize random generators. [default: 12345]         │
	│ --help                            Show this message and exit.                                    │
	╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
	╭─ Data ───────────────────────────────────────────────────────────────────────────────────────────╮
	│ --dataset                   [ucdavis-icdm19|utmobilenet21|mi  Dataset to use for modeling.       │
	│                             rage19|mirage22]                  [default: ucdavis-icdm19]          │
	│ --dataset-minpkts           [-1|10|100|1000]                  In combination with --dataset,     │
	│                                                               refines preprocessed and split     │
	│                                                               dataset to use.                    │
	│                                                               [default: -1]                      │
	│ --flowpic-dim               [32|64|1500]                      Flowpic dimension. [default: 32]   │
	│ --flowpic-block-duration    INTEGER                           Number of seconds for the head of  │
	│                                                               a flow (i.e., block) to use for a  │
	│                                                               flowpic.                           │
	│                                                               [default: 15]                      │
	│ --split-index               INTEGER                           Data split index. [default: 0]     │
	│ --train-val-split-ratio     FLOAT                             If not predefined by the selected  │
	│                                                               split, the ratio data to use for   │
	│                                                               training (rest is for validation). │
	│                                                               [default: 0.8]                     │
	│ --aug-name                  [noaug|rotate|horizontalflip|col  Name of the augmentation to use.   │
	│                             orjitter|packetloss|timeshift|ch  [default: noaug]                   │
	│                             angertt]                                                             │
	│ --no-test-leftover                                            Skip test on leftover split        │
	│                                                               (specific for ucdavis-icdm19, and  │
	│                                                               default enabled for all other      │
	│                                                               datasets).                         │
	╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
	╭─ Modeling ───────────────────────────────────────────────────────────────────────────────────────╮
	│ --method    [monolithic|xgboost]  Method to use for training. [default: monolithic]              │
	╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
	╭─ DL hyper params ────────────────────────────────────────────────────────────────────────────────╮
	│ --batch-size        INTEGER  Training batch size [default: 32]                                   │
	│ --learning-rate     FLOAT    Training learning rate. [default: 0.001]                            │
	│ --patience-steps    INTEGER  Max. number of epochs without improvement before stopping training. │
	│                              [default: 5]                                                        │
	│ --epochs            INTEGER  Number of epochs for training. [default: 50]                        │
	│ --no-dropout                 Mask dropout layers with Identity layers.                           │
	╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
	╭─ XGBoost hyper params ───────────────────────────────────────────────────────────────────────────╮
	│ --input-repr       [flowpic|pktseries]  Input representation. [default: pktseries]               │
	│ --pktseries-len    INTEGER              Number of packets (when using time series as input).     │
	│                                         [default: 10]                                            │
	╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
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
	connecting to AIM repo at: aim-repo
	created aim run hash=d5fa0dae7540485682e0869e
	artifacts folder at: aim-repo/artifacts/d5fa0dae7540485682e0869e
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
	opened log at aim-repo/artifacts/d5fa0dae7540485682e0869e/log.txt
	loaded: ./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/imc23/train_split_0.parquet
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

	network architecture
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

	epoch:   0 | train_loss: 1.484430 | train_acc:  38.5% | val_loss: 1.207108 | val_acc:  95.0% | *
	epoch:   1 | train_loss: 1.234462 | train_acc:  49.2% | val_loss: 0.782972 | val_acc:  96.0% | *
	epoch:   2 | train_loss: 1.023239 | train_acc:  56.8% | val_loss: 0.531561 | val_acc:  95.0% | *
	epoch:   3 | train_loss: 1.015719 | train_acc:  54.0% | val_loss: 0.408276 | val_acc:  95.0% | *
	epoch:   4 | train_loss: 0.952349 | train_acc:  53.5% | val_loss: 0.292073 | val_acc:  97.0% | *
	epoch:   5 | train_loss: 0.911262 | train_acc:  57.5% | val_loss: 0.359489 | val_acc:  96.0%
	epoch:   6 | train_loss: 0.920104 | train_acc:  59.8% | val_loss: 0.256316 | val_acc:  97.0% | *
	epoch:   7 | train_loss: 0.964038 | train_acc:  53.8% | val_loss: 0.233604 | val_acc:  97.0% | *
	epoch:   8 | train_loss: 0.868652 | train_acc:  59.5% | val_loss: 0.273222 | val_acc:  98.0%
	epoch:   9 | train_loss: 0.878158 | train_acc:  58.8% | val_loss: 0.191980 | val_acc:  97.0% | *
	epoch:  10 | train_loss: 0.814328 | train_acc:  59.5% | val_loss: 0.193263 | val_acc:  98.0%
	epoch:  11 | train_loss: 0.873851 | train_acc:  54.8% | val_loss: 0.179443 | val_acc:  97.0% | *
	epoch:  12 | train_loss: 0.889558 | train_acc:  57.0% | val_loss: 0.161395 | val_acc:  97.0% | *
	epoch:  13 | train_loss: 0.862803 | train_acc:  56.8% | val_loss: 0.201240 | val_acc:  98.0%
	epoch:  14 | train_loss: 0.952843 | train_acc:  51.8% | val_loss: 0.144734 | val_acc:  97.0% | *
	epoch:  15 | train_loss: 0.893776 | train_acc:  56.8% | val_loss: 0.130068 | val_acc:  98.0% | *
	epoch:  16 | train_loss: 0.834928 | train_acc:  59.2% | val_loss: 0.176018 | val_acc:  96.0%
	epoch:  17 | train_loss: 0.883773 | train_acc:  57.8% | val_loss: 0.129550 | val_acc:  98.0%
	epoch:  18 | train_loss: 0.822181 | train_acc:  59.5% | val_loss: 0.134569 | val_acc:  97.0%
	epoch:  19 | train_loss: 0.861686 | train_acc:  58.5% | val_loss: 0.117548 | val_acc:  98.0% | *
	epoch:  20 | train_loss: 0.856357 | train_acc:  60.2% | val_loss: 0.182993 | val_acc:  96.0%
	epoch:  21 | train_loss: 0.929739 | train_acc:  55.0% | val_loss: 0.144799 | val_acc:  98.0%
	epoch:  22 | train_loss: 0.806559 | train_acc:  60.8% | val_loss: 0.112128 | val_acc:  98.0% | *
	epoch:  23 | train_loss: 0.893108 | train_acc:  56.8% | val_loss: 0.110627 | val_acc:  98.0% | *
	epoch:  24 | train_loss: 0.814123 | train_acc:  62.0% | val_loss: 0.125326 | val_acc:  98.0%
	epoch:  25 | train_loss: 0.804388 | train_acc:  59.2% | val_loss: 0.119205 | val_acc:  98.0%
	epoch:  26 | train_loss: 0.868501 | train_acc:  57.8% | val_loss: 0.101536 | val_acc:  98.0% | *
	epoch:  27 | train_loss: 0.799067 | train_acc:  61.8% | val_loss: 0.105484 | val_acc:  98.0%
	epoch:  28 | train_loss: 0.842477 | train_acc:  58.0% | val_loss: 0.086314 | val_acc:  98.0% | *
	epoch:  29 | train_loss: 0.845000 | train_acc:  57.2% | val_loss: 0.134341 | val_acc:  98.0%
	epoch:  30 | train_loss: 0.731437 | train_acc:  64.2% | val_loss: 0.079428 | val_acc:  97.0% | *
	epoch:  31 | train_loss: 0.791915 | train_acc:  62.8% | val_loss: 0.090225 | val_acc:  98.0%
	epoch:  32 | train_loss: 0.816266 | train_acc:  62.2% | val_loss: 0.085245 | val_acc:  98.0%
	epoch:  33 | train_loss: 0.866204 | train_acc:  56.0% | val_loss: 0.080400 | val_acc:  98.0%
	epoch:  34 | train_loss: 0.780020 | train_acc:  62.2% | val_loss: 0.109240 | val_acc:  98.0%
	epoch:  35 | train_loss: 0.865760 | train_acc:  56.5% | val_loss: 0.076947 | val_acc:  98.0% | *
	epoch:  36 | train_loss: 0.849319 | train_acc:  58.8% | val_loss: 0.078938 | val_acc:  98.0%
	epoch:  37 | train_loss: 0.833253 | train_acc:  59.0% | val_loss: 0.102355 | val_acc:  98.0%
	epoch:  38 | train_loss: 0.815432 | train_acc:  59.8% | val_loss: 0.088092 | val_acc:  98.0%
	epoch:  39 | train_loss: 0.788713 | train_acc:  61.0% | val_loss: 0.077858 | val_acc:  97.0%
	epoch:  40 | train_loss: 0.788635 | train_acc:  61.5% | val_loss: 0.090270 | val_acc:  98.0%
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

	saving: aim-repo/artifacts/d5fa0dae7540485682e0869e/train_class_rep.csv
	saving: aim-repo/artifacts/d5fa0dae7540485682e0869e/train_conf_mtx.csv


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

	saving: aim-repo/artifacts/d5fa0dae7540485682e0869e/val_class_rep.csv
	saving: aim-repo/artifacts/d5fa0dae7540485682e0869e/val_conf_mtx.csv
	loading: ./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/imc23/test_split_human.parquet
	loading: ./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/imc23/test_split_script.parquet
				   human  script
	app                         
	youtube           20      30
	google-drive      18      30
	google-doc        15      30
	google-music      15      30
	google-search     15      30

	Test dataset human | loss: 0.981303 | acc: 68.7


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

	saving: aim-repo/artifacts/d5fa0dae7540485682e0869e/test-human_class_rep.csv
	saving: aim-repo/artifacts/d5fa0dae7540485682e0869e/test-human_conf_mtx.csv

	Test dataset script | loss: 0.142414 | acc: 95.3


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

	saving: aim-repo/artifacts/d5fa0dae7540485682e0869e/test-script_class_rep.csv
	saving: aim-repo/artifacts/d5fa0dae7540485682e0869e/test-script_conf_mtx.csv
	loaded: ./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/ucdavis-icdm19.parquet

	Test dataset train-val-leftover | loss: 0.160271 | acc: 96.2


	---test-train-val-leftover reports---

				   precision    recall  f1-score      support
	google-doc      0.952576  0.999142  0.975303  1166.000000
	google-drive    0.993827  0.915929  0.953289  1582.000000
	google-music    0.845395  0.957169  0.897817   537.000000
	google-search   0.978529  0.980108  0.979318  1860.000000
	youtube         0.966667  0.960078  0.963361  1027.000000
	accuracy        0.961925  0.961925  0.961925     0.961925
	macro avg       0.947399  0.962485  0.953818  6172.000000
	weighted avg    0.963990  0.961925  0.962142  6172.000000

				   google-doc  google-drive  google-music  google-search  youtube
	google-doc           1165             0             1              0        0
	google-drive           24          1449            90              0       19
	google-music            2             4           514              7       10
	google-search          31             1             0           1823        5
	youtube                 1             4             3             33      986

	saving: aim-repo/artifacts/d5fa0dae7540485682e0869e/test-train-val-leftover_class_rep.csv
	saving: aim-repo/artifacts/d5fa0dae7540485682e0869e/test-train-val-leftover_conf_mtx.csv
	saving: aim-repo/artifacts/d5fa0dae7540485682e0869e/params.yml
    ```


## `contralearn-and-finetune`

```
tcbench run contralearn-and-finetune --help
```

!!! info "Output"
    ```
	Usage: tcbench run contralearn-and-finetune [OPTIONS]                                                                                                                                                   
	Modeling by pre-training via constrative learning and then finetune the final classifier from the
 	pre-trained model.

	╭─ General options ────────────────────────────────────────────────────────────────────────────────╮
	│ --aim-experiment-name    TEXT     The name of the experiment for AIM tracking.                   │
	│                                   [default: contrastive-learning-and-finetune]                   │
	│ --aim-repo               PATH     AIM repository location (local folder or URL).                 │
	│                                   [default: aim-repo]                                            │
	│ --artifacts-folder       PATH     Artifacts folder. [default: aim-repo/artifacts]                │
	│ --gpu-index              TEXT     The id of the GPU to use (if training with deep learning).     │
	│                                   [default: 0]                                                   │
	│ --workers                INTEGER  Number of parallel worker for loading the data. [default: 20]  │
	│ --help                            Show this message and exit.                                    │
	╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
	╭─ Data ───────────────────────────────────────────────────────────────────────────────────────────╮
	│ --dataset                   [ucdavis-icdm19]  Dataset to use for modeling.                       │
	│                                               [default: ucdavis-icdm19]                          │
	│ --flowpic-dim               [32]              Flowpic dimension. [default: 32]                   │
	│ --flowpic-block-duration    INTEGER           Number of seconds for the head of a flow (i.e.,    │
	│                                               block) to use for a flowpic.                       │
	│                                               [default: 15]                                      │
	│ --split-index               INTEGER           Data split index. [default: 0]                     │
	╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
	╭─ General Deeplearning hyperparams ───────────────────────────────────────────────────────────────╮
	│ --batch-size    INTEGER  Training batch size [default: 32]                                       │
	│ --no-dropout             Mask dropout layers with Identity layers.                               │
	╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
	╭─ Contrastive learning hyperparams ───────────────────────────────────────────────────────────────╮
	│ --cl-aug-names               TEXT     Coma separated list of augmentations pool for contrastive  │
	│                                       learning.                                                  │
	│                                       [default: changertt,timeshift]                             │
	│ --cl-projection-layer-dim    INTEGER  The number of units in the contrastive learning projection │
	│                                       layer.                                                     │
	│                                       [default: 30]                                              │
	│ --cl-learning-rate           FLOAT    Learning rate for pretraining. [default: 0.001]            │
	│ --cl-seed                    INTEGER  Seed for contrastive learning pretraining.                 │
	│                                       [default: 12345]                                           │
	│ --cl-patience-steps          INTEGER  Max steps to wait before stopping training if the top5     │
	│                                       validation accuracy does not improve.                      │
	│                                       [default: 3]                                               │
	│ --cl-temperature             FLOAT    Temperature for InfoNCE loss. [default: 0.07]              │
	│ --cl-epochs                  INTEGER  Epochs for contrastive learning pretraining. [default: 50] │
	╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
	╭─ Finetune hyperparams ───────────────────────────────────────────────────────────────────────────╮
	│ --ft-learning-rate         FLOAT    Learning rate for finetune. [default: 0.01]                  │
	│ --ft-patience-steps        INTEGER  Max steps to wait before stopping finetune training loss     │
	│                                     does not improve.                                            │
	│                                     [default: 5]                                                 │
	│ --ft-patience-min-delta    FLOAT    Minimum decrease of training loss to be considered as        │
	│                                     improvement.                                                 │
	│                                     [default: 0.001]                                             │
	│ --ft-train-samples         INTEGER  Number of samples per-class for finetune training.           │
	│                                     [default: 10]                                                │
	│ --ft-epochs                INTEGER  Epochs for finetune training. [default: 50]                  │
	│ --ft-seed                  INTEGER  Seed for finetune training. [default: 12345]                 │
	╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
    ```

### Reference example
```
tcbench run contralearn-and-finetune \
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
	connecting to AIM repo at: aim-repo
	created aim run hash=f2e52ab2cbfe4788aa642075
	artifacts folder at: aim-repo/artifacts/f2e52ab2cbfe4788aa642075
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
	train_val_split_ratio: 0.8
	-------------------
	opened log at aim-repo/artifacts/f2e52ab2cbfe4788aa642075/log.txt
	loaded: ./envs/tcbench/lib/python3.10/site-packages/tcbench/libtcdatasets/datasets/ucdavis-icdm19/preprocessed/imc23/train_split_0.parquet
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
	saving: aim-repo/artifacts/f2e52ab2cbfe4788aa642075/best_model_weights_pretrain_split_0.pt
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


	epoch:   0 | train_loss: 1.615836 | train_acc:   8.0% | *
	epoch:   1 | train_loss: 1.537240 | train_acc:  32.0% | *
	epoch:   2 | train_loss: 1.484336 | train_acc:  40.0% | *
	epoch:   3 | train_loss: 1.439574 | train_acc:  40.0% | *
	epoch:   4 | train_loss: 1.391847 | train_acc:  42.0% | *
	epoch:   5 | train_loss: 1.352634 | train_acc:  36.0% | *
	epoch:   6 | train_loss: 1.330688 | train_acc:  36.0% | *
	epoch:   7 | train_loss: 1.281217 | train_acc:  38.0% | *
	epoch:   8 | train_loss: 1.262554 | train_acc:  40.0% | *
	epoch:   9 | train_loss: 1.238030 | train_acc:  42.0% | *
	epoch:  10 | train_loss: 1.225150 | train_acc:  46.0% | *
	epoch:  11 | train_loss: 1.198335 | train_acc:  50.0% | *
	epoch:  12 | train_loss: 1.188114 | train_acc:  58.0% | *
	epoch:  13 | train_loss: 1.175914 | train_acc:  64.0% | *
	epoch:  14 | train_loss: 1.191627 | train_acc:  64.0%
	epoch:  15 | train_loss: 1.124196 | train_acc:  64.0% | *
	epoch:  16 | train_loss: 1.111688 | train_acc:  64.0% | *
	epoch:  17 | train_loss: 1.121508 | train_acc:  66.0%
	epoch:  18 | train_loss: 1.079309 | train_acc:  68.0% | *
	epoch:  19 | train_loss: 1.061186 | train_acc:  70.0% | *
	epoch:  20 | train_loss: 1.039081 | train_acc:  72.0% | *
	epoch:  21 | train_loss: 1.043780 | train_acc:  72.0%
	epoch:  22 | train_loss: 1.026590 | train_acc:  72.0% | *
	epoch:  23 | train_loss: 0.976669 | train_acc:  76.0% | *
	epoch:  24 | train_loss: 1.016128 | train_acc:  80.0%
	epoch:  25 | train_loss: 0.983972 | train_acc:  82.0%
	epoch:  26 | train_loss: 1.009065 | train_acc:  82.0%
	epoch:  27 | train_loss: 0.929888 | train_acc:  82.0% | *
	epoch:  28 | train_loss: 0.961760 | train_acc:  82.0%
	epoch:  29 | train_loss: 0.922811 | train_acc:  82.0% | *
	epoch:  30 | train_loss: 0.963608 | train_acc:  82.0%
	epoch:  31 | train_loss: 0.952226 | train_acc:  82.0%
	epoch:  32 | train_loss: 0.917957 | train_acc:  84.0% | *
	epoch:  33 | train_loss: 0.920033 | train_acc:  84.0%
	epoch:  34 | train_loss: 0.914727 | train_acc:  84.0% | *
	epoch:  35 | train_loss: 0.873566 | train_acc:  84.0% | *
	epoch:  36 | train_loss: 0.914860 | train_acc:  84.0%
	epoch:  37 | train_loss: 0.886750 | train_acc:  84.0%
	epoch:  38 | train_loss: 0.863592 | train_acc:  84.0% | *
	epoch:  39 | train_loss: 0.875310 | train_acc:  84.0%
	epoch:  40 | train_loss: 0.897717 | train_acc:  82.0%
	epoch:  41 | train_loss: 0.875007 | train_acc:  82.0%
	epoch:  42 | train_loss: 0.840733 | train_acc:  84.0% | *
	epoch:  43 | train_loss: 0.810166 | train_acc:  84.0% | *
	epoch:  44 | train_loss: 0.819367 | train_acc:  84.0%
	epoch:  45 | train_loss: 0.823529 | train_acc:  84.0%
	epoch:  46 | train_loss: 0.813771 | train_acc:  84.0%
	epoch:  47 | train_loss: 0.846898 | train_acc:  84.0%
	epoch:  48 | train_loss: 0.804951 | train_acc:  86.0% | *
	epoch:  49 | train_loss: 0.814344 | train_acc:  86.0%
	reached max epochs
	saving: aim-repo/artifacts/f2e52ab2cbfe4788aa642075/best_model_weights_finetune_human_from_split_0.pt

	--- finetune (test) on human ---
	app
	youtube          10
	google-drive      8
	google-doc        5
	google-music      5
	google-search     5
	Name: count, dtype: int64

	Test dataset human | loss: 1.122207 | acc: 75.8


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

	saving: aim-repo/artifacts/f2e52ab2cbfe4788aa642075/test-human_class_rep.csv
	saving: aim-repo/artifacts/f2e52ab2cbfe4788aa642075/test-human_conf_mtx.csv

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

	epoch:   0 | train_loss: 1.610058 | train_acc:  20.0% | *
	epoch:   1 | train_loss: 1.551162 | train_acc:  34.0% | *
	epoch:   2 | train_loss: 1.494622 | train_acc:  44.0% | *
	epoch:   3 | train_loss: 1.432214 | train_acc:  46.0% | *
	epoch:   4 | train_loss: 1.373513 | train_acc:  52.0% | *
	epoch:   5 | train_loss: 1.349639 | train_acc:  54.0% | *
	epoch:   6 | train_loss: 1.294582 | train_acc:  56.0% | *
	epoch:   7 | train_loss: 1.281295 | train_acc:  70.0% | *
	epoch:   8 | train_loss: 1.241570 | train_acc:  72.0% | *
	epoch:   9 | train_loss: 1.181321 | train_acc:  76.0% | *
	epoch:  10 | train_loss: 1.124716 | train_acc:  78.0% | *
	epoch:  11 | train_loss: 1.144935 | train_acc:  78.0%
	epoch:  12 | train_loss: 1.103080 | train_acc:  78.0% | *
	epoch:  13 | train_loss: 1.061779 | train_acc:  78.0% | *
	epoch:  14 | train_loss: 1.030383 | train_acc:  76.0% | *
	epoch:  15 | train_loss: 1.030399 | train_acc:  76.0%
	epoch:  16 | train_loss: 0.954889 | train_acc:  76.0% | *
	epoch:  17 | train_loss: 0.955167 | train_acc:  76.0%
	epoch:  18 | train_loss: 0.959741 | train_acc:  80.0%
	epoch:  19 | train_loss: 0.896621 | train_acc:  84.0% | *
	epoch:  20 | train_loss: 0.886190 | train_acc:  92.0% | *
	epoch:  21 | train_loss: 0.909035 | train_acc:  92.0%
	epoch:  22 | train_loss: 0.888979 | train_acc:  92.0%
	epoch:  23 | train_loss: 0.860065 | train_acc:  92.0% | *
	epoch:  24 | train_loss: 0.820559 | train_acc:  92.0% | *
	epoch:  25 | train_loss: 0.800317 | train_acc:  92.0% | *
	epoch:  26 | train_loss: 0.761238 | train_acc:  92.0% | *
	epoch:  27 | train_loss: 0.759209 | train_acc:  92.0% | *
	epoch:  28 | train_loss: 0.789361 | train_acc:  92.0%
	epoch:  29 | train_loss: 0.751218 | train_acc:  92.0% | *
	epoch:  30 | train_loss: 0.743825 | train_acc:  94.0% | *
	epoch:  31 | train_loss: 0.758159 | train_acc:  94.0%
	epoch:  32 | train_loss: 0.726163 | train_acc:  92.0% | *
	epoch:  33 | train_loss: 0.700075 | train_acc:  92.0% | *
	epoch:  34 | train_loss: 0.657465 | train_acc:  92.0% | *
	epoch:  35 | train_loss: 0.656949 | train_acc:  92.0%
	epoch:  36 | train_loss: 0.671816 | train_acc:  92.0%
	epoch:  37 | train_loss: 0.675776 | train_acc:  92.0%
	epoch:  38 | train_loss: 0.621407 | train_acc:  92.0% | *
	epoch:  39 | train_loss: 0.642558 | train_acc:  92.0%
	epoch:  40 | train_loss: 0.624327 | train_acc:  92.0%
	epoch:  41 | train_loss: 0.639679 | train_acc:  92.0%
	epoch:  42 | train_loss: 0.594589 | train_acc:  92.0% | *
	epoch:  43 | train_loss: 0.617626 | train_acc:  92.0%
	epoch:  44 | train_loss: 0.597697 | train_acc:  92.0%
	epoch:  45 | train_loss: 0.598853 | train_acc:  92.0%
	epoch:  46 | train_loss: 0.629960 | train_acc:  92.0%
	epoch:  47 | train_loss: 0.596823 | train_acc:  92.0%
	run out of patience
	saving: aim-repo/artifacts/f2e52ab2cbfe4788aa642075/best_model_weights_finetune_script_from_split_0.pt

	--- finetune (test) on script ---
	app
	google-doc       20
	google-drive     20
	google-music     20
	google-search    20
	youtube          20
	Name: count, dtype: int64

	Test dataset script | loss: 0.565789 | acc: 93.0


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

	saving: aim-repo/artifacts/f2e52ab2cbfe4788aa642075/test-script_class_rep.csv
	saving: aim-repo/artifacts/f2e52ab2cbfe4788aa642075/test-script_conf_mtx.csv
	saving: aim-repo/artifacts/f2e52ab2cbfe4788aa642075/params.yml
    ```

## Limitations

Currently tcbench supports only the modeling functionalities
related to our [__IMC23 paper__](papers/imc23) and the 
parametrization of those runs is strictly limited to 
what was required for the purpose of the paper.

We are currently working for lifting those limitations so stay tuned :wink:
