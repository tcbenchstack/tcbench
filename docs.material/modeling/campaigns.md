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

	╭─ General options ───────────────────────────────────────────────────────────────────────────────╮
	│ --aim-experiment-name    TEXT     The name of the experiment for AIM tracking.                  │
	│                                   [default: augmentations-at-loading]                           │
	│ --aim-repo               PATH     AIM repository location (local folder or URL).                │
	│                                   [default: aim-repo]                                           │
	│ --artifacts-folder       PATH     Artifacts folder. [default: aim-repo/artifacts]               │
	│ --campaign-id            TEXT     A campaign id to mark all experiments.                        │
	│ --dry-run                         Show the number of experiments and then quit.                 │
	│ --gpu-index              TEXT     The id of the GPU to use (if training with deep learning).    │
	│                                   [default: 0]                                                  │
	│ --workers                INTEGER  Number of parallel worker for loading the data. [default: 20] │
	│ --seeds                  TEXT     Coma separated list of seed for experiments.                  │
	│                                   [default: 12345,42,666]                                       │
	│ --help                            Show this message and exit.                                   │
	╰─────────────────────────────────────────────────────────────────────────────────────────────────╯
	╭─ Data ──────────────────────────────────────────────────────────────────────────────────────────╮
	│ --augmentations       TEXT                                 Coma separated list of augmentations │
	│                                                            for experiments. Choices:            │
	│                                                            [noaug|rotate|horizontalflip|colorj… │
	│                                                            [default:                            │
	│                                                            noaug,rotate,horizontalflip,colorji… │
	│ --dataset             [ucdavis-icdm19|utmobilenet21|mirag  Dataset to use for modeling.         │
	│                       e19|mirage22]                        [default: ucdavis-icdm19]            │
	│ --dataset-minpkts     [-1|10|100|1000]                     In combination with --dataset,       │
	│                                                            refines preprocessed and split       │
	│                                                            dataset to use.                      │
	│                                                            [default: -1]                        │
	│ --flowpic-dims        TEXT                                 Coma separated list of flowpic       │
	│                                                            dimensions for experiments.          │
	│                                                            [default: 32,64,1500]                │
	│ --max-train-splits    INTEGER                              The maximum number of training       │
	│                                                            splits to experiment with. If -1,    │
	│                                                            use all available.                   │
	│                                                            [default: -1]                        │
	│ --split-indexes       TEXT                                 Coma separted list of split indexes  │
	│                                                            (by default all splits are used).    │
	│ --no-test-leftover                                         Skip test on leftover split          │
	│                                                            (specific for ucdavis-icdm19, and    │
	│                                                            default enabled for all other        │
	│                                                            datasets).                           │
	╰─────────────────────────────────────────────────────────────────────────────────────────────────╯
	╭─ Modeling ──────────────────────────────────────────────────────────────────────────────────────╮
	│ --method    [monolithic|xgboost]  Method to use for training. [default: monolithic]             │
	╰─────────────────────────────────────────────────────────────────────────────────────────────────╯
	╭─ DL hyper params ───────────────────────────────────────────────────────────────────────────────╮
	│ --batch-size        INTEGER  Training batch size. [default: 32]                                 │
	│ --epochs            INTEGER  Number of epochs for training. [default: 50]                       │
	│ --learning-rate     FLOAT    Training learning rate. [default: 0.001]                           │
	│ --patience-steps    INTEGER  Max. number of epochs without improvement before stopping          │
	│                              training.                                                          │
	│                              [default: 5]                                                       │
	│ --no-dropout                 Mask dropout layers with Identity layers.                          │
	╰─────────────────────────────────────────────────────────────────────────────────────────────────╯
	╭─ XGBoost hyper params ──────────────────────────────────────────────────────────────────────────╮
	│ --input-repr       TEXT     Input representation. [default: pktseries]                          │
	│ --pktseries-len    INTEGER  Number of packets (when using time series as input).                │
	│                             [default: 10,30]                                                    │
	╰─────────────────────────────────────────────────────────────────────────────────────────────────╯
	╭─ Options ───────────────────────────────────────────────────────────────────────────────────────╮
	│ --max-samples-per-class    INTEGER  Activated when --split-indexes is -1 to define how many     │
	│                                     samples to select for train+val (with a 80/20 split between │
	│                                     train and val).                                             │
	╰─────────────────────────────────────────────────────────────────────────────────────────────────╯
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

    ╭─ General options ───────────────────────────────────────────────────────────────────────────────╮
    │ --aim-experiment-name    TEXT     The name of the experiment for AIM tracking.                  │
    │                                   [default: contrastive-learning-and-finetune]                  │
    │ --aim-repo               PATH     AIM repository location (local folder or URL).                │
    │                                   [default: aim-repo]                                           │
    │ --artifacts-folder       PATH     Artifacts folder. [default: aim-repo/artifacts]               │
    │ --campaign-id            TEXT     A campaign id to mark all experiments.                        │
    │ --dry-run                         Show the number of experiments and then quit.                 │
    │ --gpu-index              TEXT     The id of the GPU to use (if training with deep learning).    │
    │                                   [default: 0]                                                  │
    │ --workers                INTEGER  Number of parallel worker for loading the data. [default: 50] │
    │ --help                            Show this message and exit.                                   │
    ╰─────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ Data ──────────────────────────────────────────────────────────────────────────────────────────╮
    │ --augmentations       TEXT     Coma separated list of augmentations. Choices:                   │
    │                                [noaug|rotate|horizontalflip|colorjitter|packetloss|changertt|t… │
    │                                [default: changertt,timeshift]                                   │
    │ --flowpic-dims        TEXT     Coma separated list of flowpic dimensions for experiments.       │
    │                                [default: 32]                                                    │
    │ --max-train-splits    INTEGER  The maximum number of training splits to experiment with. If -1, │
    │                                use all available.                                               │
    │                                [default: -1]                                                    │
    │ --split-indexes       TEXT     Coma separted list of split indexes (by default all splits are   │
    │                                used).                                                           │
    ╰─────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ Training hyperparams ──────────────────────────────────────────────────────────────────────────╮
    │ --batch-size                  INTEGER  Training batch size. [default: 32]                       │
    │ --cl-projection-layer-dims    TEXT     Coma separate list of contrastive learning projection    │
    │                                        layer dimensions.                                        │
    │                                        [default: 30]                                            │
    │ --cl-seeds                    TEXT     Coma separated list of seeds to use for contrastive      │
    │                                        learning pretraining.                                    │
    │                                        [default: 12345,1,2,3,4]                                 │
    │ --ft-seeds                    TEXT     Coma separated list of seeds to use for finetune         │
    │                                        training.                                                │
    │                                        [default: 12345,1,2,3,4]                                 │
    │ --dropout                     TEXT     Coma separated list. Choices:[enable|disable].           │
    │                                        [default: disable]                                       │
    ╰─────────────────────────────────────────────────────────────────────────────────────────────────╯
    ```

!!! info "Going beyond the scope of the submission"
	
	Considering the `tcbench` projection it self, it is clearly
    in the road map to extend the current functionalities with
    a full-fledged variety of options/controls beside adding
    more datasets and training methodologies.


    
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
	# campaign_id: 1688582575 | experiment 1/315 - time to completion 0:00:00
	##########

	experiment grid with 315 experiments
	---
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
    # campaign_id: 1688582864 | experiment 1/4 - time to completion 0:00:00
    ##########

    split_indexes (2): [0, 3]
    augmentations (2): ['noaug', 'rotate']
    flowpic_dims  (1): [32]
    seeds         (1): [12345]
    ```

### Estimated completion

Notice also the `campaign_id` and `time to completion`.

* `campaign_id` by default corresponds to the unixtime
of when the campaign is triggered. This can be
changed using the `--campaign-id` option.

* `time to completion` is an estimate of the remaning
time to complete the campaign based on the average
duration of the runs already completed. As such, it
is simply a rough estimate rather than a precise
estimate.

### Split campaigns across servers/repositories

For simplicity, the __runs are executed in sequence on a single-server environment__.
This is clearly not the best option in a multi-server environment,
neither using a sequence can likely take advantange of available GPUs/CPUs on a single-server.

It is however possible to use `run` and `campaign` commands to
manually split the workload and merge ML artifact folders and AIM repositories a posteriori.

!!! info "How to merge AIM repositories"
    
    The `aim` command line utility allows to copy run between
    repositories provided the hash of the runs to copy are known.

    ```
    Usage: aim runs cp [OPTIONS] [HASHES]...

      Copy Run data for given run hashes to destination Repo.

      Options:
        --destination TEXT  [required]
        --help              Show this message and exit.
    ```

    For instance, this is a minimal bash script to copy runs 
    ```bash
    #!/bin/bash

    SRC=/path/to/source/aim/repo
    DST=/path/to/destionation/aim/repo

    cd $SRC
    for run_hash in `aim runs --repo $SRC ls | grep -v Total | tr '\t' ' '`; do
        aim runs cp --destination $DST $run_hash
    done
    ```

## Submission campaigns commands

We report below the command used to trigger the campaigns
collected in the [ML artifacts](/modeling/exploring_artifacts/)

#### `ucdavis-icdm19/xgboost/noaugmentation-flowpic`

```
tcbench campaign augment-at-loading \
    --method xgboost \
    --augmentations noaug \
    --input-repr flowpic \
    --flowpic-dims 32,64,1500 \
    --seeds 12345,42,666 \
    --dataset ucdavis-icdm19 \
    --aim-repo ucdavis-icdm19/xgboost/noaugmentation-flowpic
    --artifacts-folder ucdavis-icdm19/xgboost/noaugmentation-flowpic/artifacts
```

!!! info "Runs grid"
    The campaigns has 45 runs

    ```
    split_indexes (5): [0, 1, 2, 3, 4]
    augmentations (1): ['noaug']
    seeds         (3): [12345, 42, 666]
    flowpic_dims  (3): [32, 64, 1500]
    ```

    In the submission we just reported results for flowpic with 32x32 resolution.

#### `ucdavis-icdm19/xgboost/noaugmentation-timeseries`

```
tcbench campaign augment-at-loading \
    --method xgboost \
    --augmentations noaug \
    --input-repr pktseries \
    --pktseries-len 10,30 \
    --seeds 12345,42,666 \
    --dataset ucdavis-icdm19 \
    --aim-repo ucdavis-icdm19/xgboost/noaugmentation-timeseries \
    --artifacts-folder ucdavis-icdm19/xgboost/noaugmentation-timeseries/artifacts
```

!!! info "Runs grid"
    The campaign has 30 runs

	```
	split_indexes (5): [0, 1, 2, 3, 4]
	augmentations (1): ['noaug']
	seeds         (3): [12345, 42, 666]
	max_n_pkts    (2): [10, 30]
	```

    In the submission we just reported results for time series with 10 packets

#### `ucdavis-icdm19/augmentation-at-loading-with-dropout`

```
tcbench campaign augment-at-loading \
    --method monolithic \
    --seeds 12345,42,666 \
    --dataset ucdavis-icdm19 \
    --aim-repo ucdavis-icdm19/augmentation-at-loading-with-dropout \
    --artifacts-folder ucdavis-icdm19/augmentation-at-loading-with-dropout/artifacts
```

!!! info "Runs grid"
    The campaign has 315 runs
	```
	split_indexes (5): [0, 1, 2, 3, 4]
	augmentations (7): ['noaug', 'rotate', 'horizontalflip', 'colorjitter', 'packetloss', 'changertt', 'timeshift']
	flowpic_dims  (3): [32, 64, 1500]
	seeds         (3): [12345, 42, 666]
	```

#### `mirage19/augmentation-at-loading-no-dropout/minpkts10`

```
tcbench campaign augment-at-loading \
    --method monolithic \
    --seeds 12345,42,666 \
    --dataset mirage19 \
    --dataset-minpkts 10 \
    --augmentations noaug \
    --no-dropout \
    --flowpic-dims 32 \
    --aim-repo mirage19/augmentation-at-loading-no-dropout/minpkts10 \
    --artifacts-folder mirage19/augmentation-at-loading-no-dropout/minpkts10/artifacts
```

!!! info "Runs grid"
    The campaign has 15 runs
	```
	split_indexes (5): [0, 1, 2, 3, 4]
	augmentations (1): ['noaug']
	flowpic_dims  (1): [32]
	seeds         (3): [12345, 42, 666]
	```

#### `mirage22/augmentation-at-loading-no-dropout/minpkts10`

```
tcbench campaign augment-at-loading \
    --method monolithic \
    --seeds 12345,42,666 \
    --dataset mirage22 \
    --dataset-minpkts 10 \
    --augmentations noaug \
    --no-dropout \
    --flowpic-dims 32 \
    --aim-repo mirage22/augmentation-at-loading-no-dropout/minpkts10 \
    --artifacts-folder mirage22/augmentation-at-loading-no-dropout/minpkts10/artifacts
```

!!! info "Runs grid"
	The campaign has 15 runs
	```
	split_indexes (5): [0, 1, 2, 3, 4]
	augmentations (1): ['noaug']
	flowpic_dims  (1): [32]
	seeds         (3): [12345, 42, 666]
	```

#### `mirage22/augmentation-at-loading-no-dropout/minpkts1000`

```
tcbench campaign augment-at-loading \
    --method monolithic \
    --seeds 12345,42,666 \
    --dataset mirage22 \
    --dataset-minpkts 10 \
    --augmentations noaug \
    --no-dropout \
    --flowpic-dims 32 \
    --aim-repo mirage22/augmentation-at-loading-no-dropout/minpkts10 \
    --artifacts-folder mirage22/augmentation-at-loading-no-dropout/minpkts10
```

!!! info "Runs grid"
	The campaign has 15 runs
	```
	split_indexes (5): [0, 1, 2, 3, 4]
	augmentations (1): ['noaug']
	flowpic_dims  (1): [32]
	seeds         (3): [12345, 42, 666]
	```

#### `utmobilenet21/augmentation-at-loading-no-dropout/minpkts10`

```
tcbench campaign augment-at-loading \
    --method monolithic \
    --seeds 12345,42,666 \
    --dataset utmobilenet21 \
    --dataset-minpkts 10 \
    --augmentations noaug \
    --no-dropout \
    --flowpic-dims 32 \
    --aim-repo utmobilenet21/augmentation-at-loading-no-dropout/minpkts10 \
    --artifacts-folder utmobilenet21/augmentation-at-loading-no-dropout/minpkts10/artifacts
```

!!! info "Runs grid"
	The campaign has 15 runs
	```
	split_indexes (5): [0, 1, 2, 3, 4]
	augmentations (1): ['noaug']
	flowpic_dims  (1): [32]
	seeds         (3): [12345, 42, 666]
	```

#### `ucdavis-icdm19/simclr-other-augmentation-pairs/colorjitter-changertt`
```
tcbench campaign contralearn-and-finetune \
    --augmentations colorjitter,changertt \
    --flowpic-dims 32 \
    --cl-seeds 12345,1,2,3,4 \
    --ft-seeds 12345,1,2,3,4 \
    --cl-projection-layer-dims 30 \
    --dropout disable \
    --aim-repo ucdavis-icdm19/simclr-other-augmentation-pairs/colorjitter-changertt \
    --artifacts-folder ucdavis-icdm19/simclr-other-augmentation-pairs/colorjitter-changertt/artifacts
```

!!! info "Runs grid"
	The campaign has 125 runs
	```
    split_indexes              (5): [0, 1, 2, 3, 4]
    contrastive learning seeds (5): [12345, 1, 2, 3, 4]
    finetune seeds             (5): [12345, 1, 2, 3, 4]
    projection layer dims      (1): [30]
    dropout                    (1): ['disable']
    flowpic dims               (1): [32]
	```
#### `ucdavis-icdm19/simclr-other-augmentation-pairs/colorjitter-packetloss`
```
tcbench campaign contralearn-and-finetune \
    --augmentations colorjitter,packetloss \
    --flowpic-dims 32 \
    --cl-seeds 12345,1,2,3,4 \
    --ft-seeds 12345,1,2,3,4 \
    --cl-projection-layer-dims 30 \
    --dropout disable \
    --aim-repo ucdavis-icdm19/simclr-other-augmentation-pairs/colorjitter-packetloss \
    --artifacts-folder ucdavis-icdm19/simclr-other-augmentation-pairs/colorjitter-packetloss/artifacts
```

!!! info "Runs grid"
	The campaign has 125 runs
	```
    split_indexes              (5): [0, 1, 2, 3, 4]
    contrastive learning seeds (5): [12345, 1, 2, 3, 4]
    finetune seeds             (5): [12345, 1, 2, 3, 4]
    projection layer dims      (1): [30]
    dropout                    (1): ['disable']
    flowpic dims               (1): [32]
	```

#### `ucdavis-icdm19/simclr-other-augmentation-pairs/colorjitter-rotate`
```
tcbench campaign contralearn-and-finetune \
    --augmentations colorjitter,rotate \
    --flowpic-dims 32 \
    --cl-seeds 12345,1,2,3,4 \
    --ft-seeds 12345,1,2,3,4 \
    --cl-projection-layer-dims 30 \
    --dropout disable \
    --aim-repo ucdavis-icdm19/simclr-other-augmentation-pairs/colorjitter-rotate \
    --artifacts-folder ucdavis-icdm19/simclr-other-augmentation-pairs/colorjitter-rotate/artifacts
```

!!! info "Runs grid"
	The campaign has 125 runs
	```
    split_indexes              (5): [0, 1, 2, 3, 4]
    contrastive learning seeds (5): [12345, 1, 2, 3, 4]
    finetune seeds             (5): [12345, 1, 2, 3, 4]
    projection layer dims      (1): [30]
    dropout                    (1): ['disable']
    flowpic dims               (1): [32]
	```

#### `ucdavis-icdm19/simclr-other-augmentation-pairs/rotate-changertt`
```
tcbench campaign contralearn-and-finetune \
    --augmentations rotate,changertt \
    --flowpic-dims 32 \
    --cl-seeds 12345,1,2,3,4 \
    --ft-seeds 12345,1,2,3,4 \
    --cl-projection-layer-dims 30 \
    --dropout disable \
    --aim-repo ucdavis-icdm19/simclr-other-augmentation-pairs/rotate-changertt \
    --artifacts-folder ucdavis-icdm19/simclr-other-augmentation-pairs/rotate-changertt/artifacts
```

!!! info "Runs grid"
	The campaign has 125 runs
	```
    split_indexes              (5): [0, 1, 2, 3, 4]
    contrastive learning seeds (5): [12345, 1, 2, 3, 4]
    finetune seeds             (5): [12345, 1, 2, 3, 4]
    projection layer dims      (1): [30]
    dropout                    (1): ['disable']
    flowpic dims               (1): [32]
	```

#### `ucdavis-icdm19/simclr-other-augmentation-pairs/rotate-packetloss`
```
tcbench campaign contralearn-and-finetune \
    --augmentations rotate,packetloss \
    --flowpic-dims 32 \
    --cl-seeds 12345,1,2,3,4 \
    --ft-seeds 12345,1,2,3,4 \
    --cl-projection-layer-dims 30 \
    --dropout disable \
    --aim-repo ucdavis-icdm19/simclr-other-augmentation-pairs/rotate-packetloss \
    --artifacts-folder ucdavis-icdm19/simclr-other-augmentation-pairs/rotate-packetloss/artifacts
```

!!! info "Runs grid"
	The campaign has 215 runs
	```
    split_indexes              (5): [0, 1, 2, 3, 4]
    contrastive learning seeds (5): [12345, 1, 2, 3, 4]
    finetune seeds             (5): [12345, 1, 2, 3, 4]
    projection layer dims      (1): [30]
    dropout                    (1): ['disable']
    flowpic dims               (1): [32]
	```

#### `ucdavis-icdm19/simclr-dropout-and-projection`

```
tcbench campaign contralearn-and-finetune \
    --augmentations changertt,timeshift \
    --flowpic-dims 32 \
    --cl-projection-layer-dims 30,84 \
    --cl-seeds 12345,1,2,3,4 \
    --ft-seeds 12345,1,2,3,4 \
    --dropout disable,enable \
    --aim-repo ucdavis-icdm19/simclr-dropout-and-projection \
    --artifacts-folder ucdavis-icdm19/simclr-dropout-and-projection/artifacts
```

!!! info "Run grid"
	The campaign has 500 runs
	```
	split_indexes              (5): [0, 1, 2, 3, 4]
	contrastive learning seeds (5): [12345, 1, 2, 3, 4]
	finetune seeds             (5): [12345, 1, 2, 3, 4]
	projection layer dims      (2): [30, 84]
	dropout                    (2): ['disable', 'enable']
	flowpic dims               (1): [32]
	```

#### `ucdavis-icdm19/augmentation-at-loading-suppress-dropout`
```
tcbench campaign augment-at-loading \
    --flowpic-dims 32,1500 \
    --seeds 12345,42,666 \
    --no-dropout \
    --aim-repo ucdavis-icdm19/simclr-dropout-and-projection \
    --artifacts-folder ucdavis-icdm19/simclr-dropout-and-projection/artifacts
```

!!! info "Run grid"
	The campaign has 210 runs
	```
	split_indexes (5): [0, 1, 2, 3, 4]
	augmentations (7): ['noaug', 'rotate', 'horizontalflip', 'colorjitter', 'packetloss', 'changertt', 'timeshift']
	flowpic_dims  (2): [32, 1500]
	seeds         (3): [12345, 42, 666]
	```


#### `ucdavis-icdm19/larger-trainset/augmentation-at-loading`

This campaign is a composition of two sub-campaigns 
stored in the same AIM repository.

The first is for augmentation at loading
```
tcbench campaign augment-at-loading \
    --seeds 6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25 \
    --split-indexes -1 \
    --method monolithic \
    --no-dropout \
    --flowpic-dims 32 \
    --aim-repo ucdavis-icdm19/larger-trainset/augmentation-at-loading \
    --artifacts-folder ucdavis-icdm19/larger-trainset/augmentation-at-loading/artifacts
```

!!! info "Run grid"
    The campaign has 140 runs
    ```
    split_indexes (1): [-1]
    augmentations (7): ['noaug', 'rotate', 'horizontalflip', 'colorjitter', 'packetloss', 'changertt', 'timeshift']
    flowpic_dims  (1): [32]
    seeds         (20): [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    ```

The second is for contrastive learning created with the following script to
manually compose 20 runs.

```bash
#!/bin/bash

CONTRALEARN_SEEDS=(32 33 34 35 6 7 8 9 10 11 12 13 14 15 16 17 18 20 43 64)
FINETUNE_SEEDS=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 22 43 64)


for idx in {0..19}; do
    tcbench run contralearn-and-finetune \
        --dataset ucdavis-icdm19 \
        --batch-size 32 \
        --flowpic-dim 32 \
        --no-dropout \
        --split-index -1 \
        --cl-projection-layer-dim 30 \
        --cl-seed ${CONTRALEARN_SEEDS[$idx]} \
        --ft-seed ${FINETUNE_SEEDS[$idx]} \
        --aim-repo ucdavis-icdm19/larger-trainset/augmentation-at-loading \
        --artifacts-folder ucdavis-icdm19/larger-trainset/augmentation-at-loading/artifacts
done
```

#### `ucdavis-icdm19-git-repo-forked`

This campaign is different from all the others
because is just [repeating]() the experiments
of the following paper ICDM19.

```
@misc{rezaei2020achieve,
title={How to Achieve High Classification Accuracy with Just a Few Labels: A Semi-supervised Approach Using Sampled Packets}, 
author={Shahbaz Rezaei and Xin Liu},
year={2020},
eprint={1812.09761},
archivePrefix={arXiv},
primaryClass={cs.NI}
}
```

The related code is available at this 
[:simple-github: repository](https://github.com/shrezaei/Semi-supervised-Learning-QUIC-)

We just did minor modifications (mostly
for changing output folders) without
affecting how to execute the code.

In other words, to generate the result run
```
python dataProcessInMemoryQUIC.py 
python pre-training.py
python re-training.py
```

All results are collected in an output `/artifacts` folder
(but now AIM repository is generated).
