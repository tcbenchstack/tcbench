---
title: Campaigns
icon: simple/docsdotrs
---

Individual modeling campaings can be triggered
from the subcommand `campaign` sub-command.

```
tcbench campaign --help
```

!!! info "Output"
	```
	 Usage: tcbench campaign [OPTIONS] COMMAND [ARGS]...

	 Triggers a modeling campaign.

	╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
	│ --help      Show this message and exit.                                                          │
	╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
	╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────╮
	│ augment-at-loading        Modeling by applying data augmentation when loading the training set.  │
	│ contralearn-and-finetune  Modeling by pre-training via constrative learning and then finetune    │
	│                           the final classifier from the pre-trained model.                       │
	╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
	```

The `campaign` sub-command is an __"opinionated version"__ of `run` sub-command.
Meaning, is currently targetting only the needs for the campaigns which
are part of the submission. So, the options exposed by the `campaign` sub-commands
are a selected subset of the one options available for the related `run` sub-commands.

For `augment-at-loading` campaign supports the following options
```
tcbench campaign augment-at-loading --help
```

!!! info "Output"
	```
                                                                             
	Usage: tcbench campaign augment-at-loading [OPTIONS]                                                                                                                                                    
	Modeling by applying data augmentation when loading the training set.

	╭─ General options ────────────────────────────────────────────────────────────────────────────────╮
	│ --aim-experiment-name    TEXT     The name of the experiment for AIM tracking.                   │
	│                                   [default: augmentations-at-loading]                            │
	│ --aim-repo               PATH     AIM repository location (local folder or URL).                 │
	│                                   [default: aim-repo]                                            │
	│ --artifacts-folder       PATH     Artifacts folder. [default: aim-repo/artifacts]                │
	│ --campaign-id            TEXT     A campaign id to mark all experiments.                         │
	│ --dry-run                         Show the number of experiments and then quit.                  │
	│ --gpu-index              TEXT     The id of the GPU to use (if training with deep learning).     │
	│                                   [default: 0]                                                   │
	│ --workers                INTEGER  Number of parallel worker for loading the data. [default: 20]  │
	│ --seeds                  TEXT     Coma separated list of seed for experiments.                   │
	│                                   [default: 12345,42,666]                                        │
	│ --help                            Show this message and exit.                                    │
	╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
	╭─ Data ───────────────────────────────────────────────────────────────────────────────────────────╮
	│ --augmentations       TEXT                                  Coma separated list of augmentations │
	│                                                             for experiments. Choices:            │
	│                                                             [noaug|rotate|horizontalflip|colorj… │
	│                                                             [default:                            │
	│                                                             noaug,rotate,horizontalflip,colorji… │
	│ --dataset             [ucdavis-icdm19|utmobilenet21|mirage  Dataset to use for modeling.         │
	│                       19|mirage22]                          [default: ucdavis-icdm19]            │
	│ --dataset-minpkts     [-1|10|100|1000]                      In combination with --dataset,       │
	│                                                             refines preprocessed and split       │
	│                                                             dataset to use.                      │
	│                                                             [default: -1]                        │
	│ --flowpic-dims        TEXT                                  Coma separated list of flowpic       │
	│                                                             dimensions for experiments.          │
	│                                                             [default: 32,64,1500]                │
	│ --max-train-splits    INTEGER                               The maximum number of training       │
	│                                                             splits to experiment with. If -1,    │
	│                                                             use all available.                   │
	│                                                             [default: -1]                        │
	│ --split-indexes       TEXT                                  Coma separted list of split indexes  │
	│                                                             (by default all splits are used).    │
	│ --no-test-leftover                                          Skip test on leftover split          │
	│                                                             (specific for ucdavis-icdm19, and    │
	│                                                             default enabled for all other        │
	│                                                             datasets).                           │
	╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
	╭─ Modeling ───────────────────────────────────────────────────────────────────────────────────────╮
	│ --method    [monolithic|xgboost]  Method to use for training. [default: monolithic]              │
	╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
	╭─ DL hyper params ────────────────────────────────────────────────────────────────────────────────╮
	│ --batch-size        INTEGER  Training batch size. [default: 32]                                  │
	│ --epochs            INTEGER  Number of epochs for training. [default: 50]                        │
	│ --learning-rate     FLOAT    Training learning rate. [default: 0.001]                            │
	│ --patience-steps    INTEGER  Max. number of epochs without improvement before stopping training. │
	│                              [default: 5]                                                        │
	│ --no-dropout                 Mask dropout layers with Identity layers.                           │
	╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
	╭─ XGBoost hyper params ───────────────────────────────────────────────────────────────────────────╮
	│ --input-repr       TEXT     Input representation. [default: pktseries]                           │
	│ --pktseries-len    INTEGER  Number of packets (when using time series as input).                 │
	│                             [default: 10,30]                                                     │
	╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
	╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
	│ --max-samples-per-class    INTEGER  Activated when --split-indexes is -1 to define how many      │
	│                                     samples to select for train+val (with a 80/20 split between  │
	│                                     train and val).                                              │
	╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
	```

For `contralearn-and-finetune` campaign supports the following options
```
tcbench campaign contralearn-and-finetune --help
```

!!! info "Output"
```
tcbench campaign contralearn-and-finetune --help
```

!!! info "Output"
    ```
	 Usage: tcbench campaign contralearn-and-finetune [OPTIONS]

	 Modeling by pre-training via constrative learning and then finetune the final classifier from the
	 pre-trained model.

	╭─ General options ────────────────────────────────────────────────────────────────────────────────╮
	│ --aim-experiment-name    TEXT     The name of the experiment for AIM tracking.                   │
	│                                   [default: contrastive-learning-and-finetune]                   │
	│ --aim-repo               PATH     AIM repository location (local folder or URL).                 │
	│                                   [default: aim-repo]                                            │
	│ --artifacts-folder       PATH     Artifacts folder. [default: aim-repo/artifacts]                │
	│ --campaign-id            TEXT     A campaign id to mark all experiments.                         │
	│ --dry-run                         Show the number of experiments and then quit.                  │
	│ --gpu-index              TEXT     The id of the GPU to use (if training with deep learning).     │
	│                                   [default: 0]                                                   │
	│ --workers                INTEGER  Number of parallel worker for loading the data. [default: 50]  │
	│ --help                            Show this message and exit.                                    │
	╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
	╭─ Data ───────────────────────────────────────────────────────────────────────────────────────────╮
	│ --augmentations            TEXT     Coma separated list of augmentations. Choices:               │
	│                                     [noaug|rotate|horizontalflip|colorjitter|packetloss|changer… │
	│                                     [default: changertt,timeshift]                               │
	│ --flowpic-dims             TEXT     Coma separated list of flowpic dimensions for experiments.   │
	│                                     [default: 32]                                                │
	│ --max-train-splits         INTEGER  The maximum number of training splits to experiment with. If │
	│                                     -1, use all available.                                       │
	│                                     [default: -1]                                                │
	│ --split-indexes            TEXT     Coma separted list of split indexes (by default all splits   │
	│                                     are used).                                                   │
	│ --train-val-split-ratio    FLOAT    If not predefined by the selected split, the ratio data to   │
	│                                     use for training (rest is for validation).                   │
	│                                     [default: 0.8]                                               │
	╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
	╭─ Training hyperparams ───────────────────────────────────────────────────────────────────────────╮
	│ --batch-size                  INTEGER  Training batch size. [default: 32]                        │
	│ --cl-projection-layer-dims    TEXT     Coma separate list of contrastive learning projection     │
	│                                        layer dimensions.                                         │
	│                                        [default: 30]                                             │
	│ --cl-seeds                    TEXT     Coma separated list of seeds to use for contrastive       │
	│                                        learning pretraining.                                     │
	│                                        [default: 12345,1,2,3,4]                                  │
	│ --ft-seeds                    TEXT     Coma separated list of seeds to use for finetune          │
	│                                        training.                                                 │
	│                                        [default: 12345,1,2,3,4]                                  │
	│ --dropout                     TEXT     Coma separated list. Choices:[enabled|disabled].          │
	│                                        [default: disabled]                                       │
	╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
    ```

## Campaign composition and progress


As mentioned, campaigns are essentially just an array of runs.


### Using `--dry-run`

The `--dry-run` option allows to
verify the composition of a campaign.

For instance
```
 tcbench campaign augment-at-loading --dry-run --method monolithic
```

!!! info "Output"
	```
	##########
	# campaign_id: 1696169422 | run 1/315 - time to completion 0:00:00
	##########

	split_indexes (5): [0, 1, 2, 3, 4]
	augmentations (7): ['noaug', 'rotate', 'horizontalflip', 'colorjitter', 'packetloss', 'changertt', 'timeshift']
	flowpic_dims  (3): [32, 64, 1500]
	seeds         (3): [12345, 42, 666]
	```

From the output we can see that the 
default configuration for an `augment-at-loading` campaign
corresponds to 315 runs = 5 splits x 7 augmentations x 3 flowpic dimensions x 3 seeds.

This can be adapted based on the campaign sub-commands options.

For instance
```
tcbench campaign augment-at-loading \
	--dry-run \
	--method monolithic \
    --split-indexes 0,3 \
    --seeds 12345 \
    --flowpic-dims 32 \
    --augmentations noaug,rotate
```

!!! info "Output"
    ```
	##########
	# campaign_id: 1696169491 | run 1/4 - time to completion 0:00:00
	##########

	split_indexes (2): [0, 3]
	augmentations (2): ['noaug', 'rotate']
	flowpic_dims  (1): [32]
	seeds         (1): [12345]
    ```

### Campaign id

Notice also how a campaign is associated to a `campaign_id` which by default
is the unixtime when the campaign is launched. This can be changed using
the `--campaign-id` options

```
tcbench campaign augment-at-loading \
    --dry-run \
    --method monolithic \
    --split-indexes 0,3 \
    --seeds 12345 \
    --flowpic-dims 32 \
    --augmentations noaug,rotate \
	--campaign-id my-wonderful-campaign
```

!!! info "Output"
	```
	##########
	# campaign_id: my-wonderful-campaign | run 1/4 - time to completion 0:00:00
	##########

	split_indexes (2): [0, 3]
	augmentations (2): ['noaug', 'rotate']
	flowpic_dims  (1): [32]
	seeds         (1): [12345]
	```


### Estimated completion

When a campaign is running, the console output
reports a status update about the number of runs
left and an estimation of the time to complete
the whole campaign.

This estimation is based on the average
duration of the runs already completed. As such, it
is simply a rough value.

### Multi-server / Multi-GPU

The runs composing a campaign are run sequentially. 
This is clearly suboptimal in a multi-server multi-gpu environment
especially if individual runs are not very computational intensive.

At this stage tcbench does not offer a mechanisms to distribute
jobs across multiple servers. However, you can launch separate
campaigns and then merge their output using the [`aimrepo merge`](/tcbench/modeling/aim_repos/aimrepo_subcmd/#merge-repositories) subcommand.

!!! tip "AIM remote server"

	AIM version 3.17.4 has the ability to spawn a 
    [remote tracking server](https://aimstack.readthedocs.io/en/v3.17.5/using/remote_tracking.html)
    but the functionality is still under development.
    
